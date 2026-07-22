#!/usr/bin/env python
"""Validate raw SMPL-X/camera geometry after LHM's image crop and resize.

Examples (run from LHM/):
  python tools/validate_lhm_geometry.py --dataset 4ddress --root .datasets/4d-dress \
      --sample-id 00123_Outer/00123_Outer_Take8_00124
  python tools/validate_lhm_geometry.py --dataset thuman --root .datasets/THuman --sample-id 0000
  python tools/validate_lhm_geometry.py --dataset huge100k --root .datasets/HuGe100K --sample-id identity_name

HuGe100K's ``smpl_params`` scale component is deliberately ignored, matching
the requested data convention.  This utility validates geometry only; it does
not load an LHM checkpoint or render predicted Gaussians.
"""

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch
import smplx

from LHM.datasets.dress4d_lhm import (
    ASPECT_HW,
    MAX_TGT_SIZE,
    _calc_tgt_hw,
    _center_crop_by_mask,
    _resize_keepaspect,
)


def lhm_crop(image_bgr, mask, intrinsic, target_width, multiply=16,
             return_transform=False):
    """Same image/mask/K transform used by Dress4DLHMDataset.

    When ``return_transform`` is true, also return the exact image-space
    affine transform used by the dataset's K update.  This makes it possible
    to test that projecting with the updated K is equivalent to projecting in
    the raw image and then applying the crop transform.
    """
    target_h, target_w = _calc_tgt_hw(ASPECT_HW, target_width, multiply)
    raw_h, raw_w = image_bgr.shape[:2]
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mask = mask.astype(np.float32)
    image = image * mask[..., None] + (1.0 - mask[..., None])
    k = intrinsic.astype(np.float64).copy()
    image, ratio = _resize_keepaspect(image, MAX_TGT_SIZE)
    mask, _ = _resize_keepaspect(mask, MAX_TGT_SIZE)
    k[0, 0] *= ratio; k[0, 2] *= ratio
    k[1, 1] *= ratio; k[1, 2] *= ratio
    image, mask, off_x, off_y = _center_crop_by_mask(image, mask, k[0, 2], k[1, 2], ASPECT_HW)
    k[0, 2] -= off_x; k[1, 2] -= off_y
    h, w = image.shape[:2]
    sy, sx = target_h / h, target_w / w
    image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    k[0, 0] *= sx; k[0, 2] *= sx
    k[1, 1] *= sy; k[1, 2] *= sy
    if not np.allclose(k[:2, 2], [target_w / 2, target_h / 2], rtol=0, atol=1e-4):
        raise ValueError(f'LHM crop did not center principal point: {k[:2, 2]}')
    if not return_transform:
        return image, mask > .5, k
    transform = {
        'raw_hw': (raw_h, raw_w),
        # These are deliberately the same factors used to update K.
        'pre_resize_scale': float(ratio),
        'crop_offset_xy': (int(off_x), int(off_y)),
        'crop_hw': (int(h), int(w)),
        'final_resize_scale_xy': (float(sx), float(sy)),
        'output_hw': (target_h, target_w),
    }
    return image, mask > .5, k, transform


def find_huge_param(root, sample_id):
    matches = list(Path(root).glob(f'**/param/{sample_id}.npy'))
    if len(matches) != 1:
        raise FileNotFoundError(f'expected one **/param/{sample_id}.npy, found {len(matches)}')
    return matches[0]


def load_sample(dataset, root, sample_id):
    root = Path(root)
    if dataset == '4ddress':
        directory = root / sample_id
        with (directory / 'camera.json').open() as handle:
            cameras = json.load(handle)
        with next((directory / 'smplx').glob('*.pkl')).open('rb') as handle:
            params = pickle.load(handle)
        image = lambda v: directory / 'image' / f'{v}.png'
        mask = lambda v: directory / 'mask' / f'{v}.png'
        return cameras, params, image, mask, True
    if dataset == 'thuman':
        directory = root / 'process' / sample_id
        with (directory / 'camera.json').open() as handle:
            cameras = json.load(handle)
        with (root / 'THuman2.1_smplx' / sample_id / 'smplx_param.pkl').open('rb') as handle:
            params = pickle.load(handle)
        image = lambda v: directory / 'image' / f'{v}.png'
        mask = lambda v: directory / 'mask' / f'{v}.png'
        return cameras, params, image, mask, False

    param_path = find_huge_param(root, sample_id)
    data = np.load(param_path, allow_pickle=True).item()
    cameras = {}
    for view, pose in enumerate(data['poses']):
        extrinsic, intr = np.asarray(pose[0]), np.asarray(pose[1])
        cameras[f'{view:02d}'] = {
            'R': extrinsic[:3, :3], 'T': extrinsic[:3, 3],
            'K': np.array([[intr[0], 0, intr[2]], [0, intr[1], intr[3]], [0, 0, 1]], np.float32),
        }
    vector = torch.as_tensor(data['smpl_params'], dtype=torch.float32).reshape(1, -1)
    # scale is intentionally ignored.
    _, transl, orient, body, betas, left, right, jaw, leye, reye, expr = torch.split(
        vector, [1, 3, 3, 63, 10, 45, 45, 3, 3, 3, 10], dim=1)
    params = {'transl': transl, 'global_orient': orient, 'body_pose': body,
              'betas': betas, 'left_hand_pose': left, 'right_hand_pose': right,
              'jaw_pose': jaw, 'leye_pose': leye, 'reye_pose': reye, 'expression': expr}
    image_dir = param_path.parent.parent / 'images' / sample_id
    image = lambda v: image_dir / f'view_{v}.png'
    return cameras, params, image, None, False


def tensor_param(params, name, size, device):
    return torch.as_tensor(params[name], dtype=torch.float32, device=device).reshape(1, size)


def forward_vertices(params, pca, model_path, device):
    expression = np.asarray(params['expression']).reshape(-1)
    model = smplx.create(model_path, model_type='smplx', gender='neutral', num_betas=10,
                         num_expression_coeffs=len(expression), use_pca=pca,
                         num_pca_comps=12, flat_hand_mean=not pca).to(device).eval()
    with torch.no_grad():
        output = model(
            betas=tensor_param(params, 'betas', 10, device),
            global_orient=tensor_param(params, 'global_orient', 3, device),
            body_pose=tensor_param(params, 'body_pose', 63, device),
            transl=tensor_param(params, 'transl', 3, device),
            left_hand_pose=tensor_param(params, 'left_hand_pose', 12 if pca else 45, device),
            right_hand_pose=tensor_param(params, 'right_hand_pose', 12 if pca else 45, device),
            jaw_pose=tensor_param(params, 'jaw_pose', 3, device),
            leye_pose=tensor_param(params, 'leye_pose', 3, device),
            reye_pose=tensor_param(params, 'reye_pose', 3, device),
            expression=tensor_param(params, 'expression', len(expression), device),
        )
    return output.vertices[0].cpu().numpy(), model.faces.astype(np.int32)


def silhouette(vertices, faces, camera, intrinsic, height, width):
    uv, z = project_vertices(vertices, camera, intrinsic)
    out = np.zeros((height, width), np.uint8)
    for face in faces:
        if np.any(z[face] <= 1e-6):
            continue
        triangle = np.rint(uv[face]).astype(np.int32)
        cv2.fillConvexPoly(out, triangle, 1)
    return out.astype(bool)


def project_vertices(vertices, camera, intrinsic):
    """Project vertices in the raw or cropped image coordinate system."""
    r = np.asarray(camera['R'], np.float32)
    t = np.asarray(camera['T'], np.float32)
    points = vertices @ r.T + t
    uvw = points @ np.asarray(intrinsic, np.float32).T
    uv = uvw[:, :2] / np.maximum(uvw[:, 2:3], 1e-8)
    return uv, points[:, 2]


def apply_lhm_crop_transform(uv, transform, pixel_center=False):
    """Map raw projected pixels to the final LHM crop coordinates.

    ``pixel_center=False`` reproduces the K update in the dataset exactly.
    ``pixel_center=True`` additionally applies OpenCV's half-pixel resize
    convention, which quantifies the small sub-pixel discrepancy omitted by
    the conventional ``K *= scale`` camera update.
    """
    result = np.asarray(uv, np.float64).copy()
    pre_scale = transform['pre_resize_scale']
    sx, sy = transform['final_resize_scale_xy']
    off_x, off_y = transform['crop_offset_xy']
    if pixel_center:
        result[:, 0] = (result[:, 0] + 0.5) * pre_scale - 0.5
        result[:, 1] = (result[:, 1] + 0.5) * pre_scale - 0.5
    else:
        result *= pre_scale
    result[:, 0] -= off_x
    result[:, 1] -= off_y
    if pixel_center:
        result[:, 0] = (result[:, 0] + 0.5) * sx - 0.5
        result[:, 1] = (result[:, 1] + 0.5) * sy - 0.5
    else:
        result[:, 0] *= sx
        result[:, 1] *= sy
    return result


def projection_errors(vertices, camera, raw_k, cropped_k, transform):
    """Compare raw-projection-plus-crop against direct cropped-K projection."""
    raw_uv, depth = project_vertices(vertices, camera, raw_k)
    cropped_uv, _ = project_vertices(vertices, camera, cropped_k)
    valid = depth > 1e-6
    expected_uv = apply_lhm_crop_transform(raw_uv[valid], transform)
    pixel_center_uv = apply_lhm_crop_transform(
        raw_uv[valid], transform, pixel_center=True
    )

    def summary(lhs, rhs):
        distances = np.linalg.norm(lhs - rhs, axis=1)
        return {
            'mean_px': float(distances.mean()),
            'max_px': float(distances.max()),
            'p99_px': float(np.quantile(distances, 0.99)),
        }

    return {
        # This must be near numerical zero. A non-zero value is a K-update bug.
        'dataset_k_affine_error': summary(expected_uv, cropped_uv[valid]),
        # This reports only OpenCV resize pixel-centre convention drift.
        'opencv_pixel_center_error': summary(pixel_center_uv, cropped_uv[valid]),
    }


def draw_contour(image, mask, color):
    """Draw a one-pixel mask boundary without changing the underlying image."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    return cv2.drawContours(image, contours, -1, color, 1, lineType=cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=('4ddress', 'thuman', 'huge100k'), required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--sample-id', required=True)
    parser.add_argument('--model-path', default='./pretrained_models/human_model_files')
    parser.add_argument('--output-dir', default='./outputs/geometry_check')
    parser.add_argument('--render-width', type=int, default=384)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    cameras, params, image_path, mask_path, pca = load_sample(args.dataset, args.root, args.sample_id)
    vertices, faces = forward_vertices(params, pca, args.model_path, args.device)
    output = Path(args.output_dir) / args.dataset / args.sample_id
    output.mkdir(parents=True, exist_ok=True)
    raw_output = output / 'raw'
    raw_output.mkdir(exist_ok=True)
    rows = []
    for index in range(24):
        view = f'{index:02d}'
        bgr = cv2.imread(str(image_path(view)))
        if bgr is None:
            raise FileNotFoundError(image_path(view))
        # Keep this threshold identical to the LHM dataset loader.
        raw_mask = (cv2.imread(str(mask_path(view)), cv2.IMREAD_GRAYSCALE) > 127
                    if mask_path else ~np.all(bgr >= 245, axis=-1))
        raw_k = np.asarray(cameras[view]['K'], dtype=np.float32)
        image, gt_mask, k, transform = lhm_crop(
            bgr, raw_mask, raw_k, args.render_width, return_transform=True
        )
        smpl_mask = silhouette(vertices, faces, cameras[view], k, image.shape[0], image.shape[1])
        errors = projection_errors(vertices, cameras[view], raw_k, k, transform)
        union = np.logical_or(gt_mask, smpl_mask).sum()
        iou = float(np.logical_and(gt_mask, smpl_mask).sum() / union) if union else 1.0
        # Red is the cropped GT mask boundary; green is the projected SMPL-X
        # boundary.  Coincident contours show as green because it is drawn last.
        overlay = draw_contour((image * 255).astype(np.uint8), gt_mask, (255, 0, 0))
        overlay = draw_contour(overlay, smpl_mask, (0, 255, 0))
        cv2.imwrite(str(output / f'{view}.png'), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        raw_image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        raw_overlay = draw_contour(raw_image, raw_mask, (255, 0, 0))
        raw_smpl_mask = silhouette(vertices, faces, cameras[view], raw_k, *bgr.shape[:2])
        raw_overlay = draw_contour(raw_overlay, raw_smpl_mask, (0, 255, 0))
        cv2.imwrite(str(raw_output / f'{view}.png'), cv2.cvtColor(raw_overlay, cv2.COLOR_RGB2BGR))
        rows.append({
            'view': view,
            'silhouette_iou': iou,
            'K': k.tolist(),
            'crop_transform': transform,
            'projection_error': errors,
        })
    affine_means = [row['projection_error']['dataset_k_affine_error']['mean_px'] for row in rows]
    affine_maxes = [row['projection_error']['dataset_k_affine_error']['max_px'] for row in rows]
    summary = {
        'dataset': args.dataset,
        'sample_id': args.sample_id,
        'huge_scale_used': False if args.dataset == 'huge100k' else None,
        'mean_silhouette_iou': float(np.mean([row['silhouette_iou'] for row in rows])),
        'mean_dataset_k_affine_error_px': float(np.mean(affine_means)),
        'max_dataset_k_affine_error_px': float(np.max(affine_maxes)),
        'views': rows,
    }
    with (output / 'summary.json').open('w') as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
