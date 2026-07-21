"""Track-metadata adapter for THuman and HuGe100K evaluation.

The Track prepare step only indexes raw samples and predicts a face box.  This
dataset deliberately keeps images, cameras and SMPL-X parameters in their raw
dataset locations.  During eval it renders every one of the 24 calibrated
views from the fixed front source view ``00``.
"""

import json
import os
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch

from .base import BaseDataset
from .dress4d_lhm import MAX_TGT_SIZE, _load_view


class StaticHumanLHMDataset(BaseDataset):
    """LHM evaluation adapter for the static THuman and HuGe100K datasets."""

    def __init__(self, root_dirs, meta_path, raw_data_dir='', source_image_res=512,
                 render_image=None, multiply=16, max_tgt_size=MAX_TGT_SIZE,
                 enlarge_ratio=(1., 1.), eval_all_views=False, src_head_size=112,
                 **kwargs):
        super().__init__(root_dirs, meta_path)
        self.root_dirs = [root_dirs] if isinstance(root_dirs, str) else list(root_dirs)
        self.raw_data_dir = Path(raw_data_dir)
        self.source_image_res = source_image_res
        self.render_image_res = (render_image or {}).get('high', (render_image or {}).get('low', 384))
        self.multiply, self.max_tgt_size = multiply, max_tgt_size
        self.enlarge_ratio, self.eval_all_views = tuple(enlarge_ratio), eval_all_views
        self.src_head_size = src_head_size
        with open(os.path.join(self.root_dirs[0], 'label', 'dataset_meta.json')) as handle:
            self.meta = json.load(handle)
        self.dataset_kind = self.meta['dataset_kind']
        self.source_view = self.meta['source_views'][0]
        self.target_groups = self.meta['target_view_groups']
        self.all_views = [f'{index:02d}' for index in range(self.meta['view_count'])]
        selected = kwargs.get('eval_sample_ids')
        if selected is not None:
            entries = list(self.uids)
            available = {entry.split('#', 1)[0] for entry in entries}
            missing = [uid for uid in selected if uid not in available]
            if missing:
                for root in self.root_dirs:
                    path = Path(root) / 'label' / 'train_list.json'
                    if path.is_file():
                        with path.open() as handle:
                            entries.extend(json.load(handle))
                available = {entry.split('#', 1)[0] for entry in entries}
            missing = [uid for uid in selected if uid not in available]
            if missing:
                raise ValueError(f'selected sample ids are not in prepared metadata: {missing[:10]}')
            selected_set = set(selected)
            self.uids = list(dict.fromkeys(
                entry for entry in entries if entry.split('#', 1)[0] in selected_set
            ))
        if eval_all_views:
            self.uids = list(dict.fromkeys(entry.split('#', 1)[0] for entry in self.uids))

    @staticmethod
    def _parse_entry(entry):
        uid, separator, group = entry.partition('#')
        return uid, int(group[1:]) if separator else 0

    @staticmethod
    def _c2w(r, t):
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = np.asarray(r, dtype=np.float32).reshape(3, 3)
        w2c[:3, 3] = np.asarray(t, dtype=np.float32).reshape(3)
        return torch.from_numpy(np.linalg.inv(w2c).astype(np.float32))

    @staticmethod
    def _as_param(data, key, shape):
        return torch.as_tensor(data[key], dtype=torch.float32).reshape(shape)

    def _face_bbox(self, uid):
        for root in self.root_dirs:
            path = Path(root) / 'face_bbox' / f'{uid}.json'
            if path.is_file():
                with path.open() as handle:
                    return json.load(handle)['bbox']
        return None

    def _head_crop(self, source_path, uid):
        bbox = self._face_bbox(uid)
        if bbox is None:
            raise FileNotFoundError(f'missing face bbox for {uid}; rerun Track prepare without --skip-face-bbox')
        image = cv2.imread(str(source_path))
        if image is None:
            raise FileNotFoundError(source_path)
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f'invalid face bbox for {uid}: {bbox}')
        crop = cv2.resize(image[y1:y2, x1:x2], (self.src_head_size, self.src_head_size),
                          interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        return torch.from_numpy(rgb).permute(2, 0, 1)

    @staticmethod
    def _pad_expression(expression):
        """LHM-MINI uses 100 FLAME coefficients; raw static sets provide 10."""
        expression = torch.as_tensor(expression, dtype=torch.float32).reshape(-1)
        if expression.numel() > 100:
            raise ValueError(f'expression has {expression.numel()} values; expected at most 100')
        return torch.nn.functional.pad(expression, (0, 100 - expression.numel()))

    def _thuman(self, uid):
        sample = self.raw_data_dir / 'process' / uid
        with (sample / 'camera.json').open() as handle:
            cameras = json.load(handle)
        with (self.raw_data_dir / 'THuman2.1_smplx' / uid / 'smplx_param.pkl').open('rb') as handle:
            params = pickle.load(handle)
        image = lambda view: sample / 'image' / f'{view}.png'
        mask = lambda view: sample / 'mask' / f'{view}.png'
        camera = lambda view: (np.asarray(cameras[view]['K']), cameras[view]['R'], cameras[view]['T'])
        packed = {
            'root_pose': self._as_param(params, 'global_orient', (3,)),
            'body_pose': self._as_param(params, 'body_pose', (21, 3)),
            'jaw_pose': self._as_param(params, 'jaw_pose', (3,)),
            'leye_pose': self._as_param(params, 'leye_pose', (3,)),
            'reye_pose': self._as_param(params, 'reye_pose', (3,)),
            'lhand_pose': self._as_param(params, 'left_hand_pose', (45,)),
            'rhand_pose': self._as_param(params, 'right_hand_pose', (45,)),
            'expr': self._pad_expression(params['expression']),
            'trans': self._as_param(params, 'transl', (3,)),
            'betas': self._as_param(params, 'betas', (10,)),
        }
        return image, mask, camera, packed, False

    def _huge100k(self, uid):
        matches = list(self.raw_data_dir.glob(f'**/param/{uid}.npy'))
        if len(matches) != 1:
            raise FileNotFoundError(f'expected exactly one **/param/{uid}.npy, found {len(matches)}')
        param_path = matches[0]
        data = np.load(param_path, allow_pickle=True).item()
        vector = torch.as_tensor(data['smpl_params'], dtype=torch.float32).reshape(1, -1)
        # The leading HuGe scale is intentionally not used: neither the
        # existing renderer nor the requested training/eval convention applies it.
        _, transl, orient, body, betas, left, right, jaw, leye, reye, expression = torch.split(
            vector, [1, 3, 3, 63, 10, 45, 45, 3, 3, 3, 10], dim=1)
        image_dir = param_path.parent.parent / 'images' / uid

        def camera(view):
            extrinsic, intrinsic = data['poses'][int(view)]
            extrinsic, intrinsic = np.asarray(extrinsic), np.asarray(intrinsic).reshape(-1)
            k = np.array([[intrinsic[0], 0, intrinsic[2]], [0, intrinsic[1], intrinsic[3]], [0, 0, 1]], np.float32)
            return k, extrinsic[:3, :3], extrinsic[:3, 3]

        packed = {
            'root_pose': orient.reshape(3), 'body_pose': body.reshape(21, 3),
            'jaw_pose': jaw.reshape(3), 'leye_pose': leye.reshape(3), 'reye_pose': reye.reshape(3),
            'lhand_pose': left.reshape(45), 'rhand_pose': right.reshape(45),
            'expr': self._pad_expression(expression), 'trans': transl.reshape(3), 'betas': betas.reshape(10),
        }
        return lambda view: image_dir / f'view_{view}.png', lambda view: None, camera, packed, True

    def inner_get_item(self, idx):
        entry = self.uids[idx]
        uid, group_index = self._parse_entry(entry)
        if self.dataset_kind == 'thuman':
            image_path, mask_path, camera, params, white_bg = self._thuman(uid)
        elif self.dataset_kind == 'huge100k':
            image_path, mask_path, camera, params, white_bg = self._huge100k(uid)
        else:
            raise ValueError(f'unsupported static dataset kind: {self.dataset_kind}')
        target_views = self.all_views if self.eval_all_views else self.target_groups[group_index]
        source_k, source_r, source_t = camera(self.source_view)
        source_image, _, source_intr = _load_view(
            str(image_path(self.source_view)), str(mask_path(self.source_view)) if mask_path(self.source_view) else '',
            source_k, self.source_image_res, self.multiply, max_tgt_size=self.max_tgt_size,
            enlarge_ratio=self.enlarge_ratio, infer_white_background=white_bg)
        images, masks, intrs, c2ws = [], [], [], []
        for view in target_views:
            k, r, t = camera(view)
            image, mask, intr = _load_view(
                str(image_path(view)), str(mask_path(view)) if mask_path(view) else '', k,
                self.render_image_res, self.multiply, max_tgt_size=self.max_tgt_size,
                enlarge_ratio=self.enlarge_ratio, infer_white_background=white_bg)
            images.append(image); masks.append(mask); intrs.append(intr); c2ws.append(self._c2w(r, t))
        count = len(target_views)
        smplx_params = {
            key: value.unsqueeze(0).expand(count, *value.shape).clone()
            for key, value in params.items() if key != 'betas'
        }
        smplx_params['betas'] = params['betas']
        return {
            'src_images': source_image.unsqueeze(0),
            'source_head_rgbs': self._head_crop(image_path(self.source_view), uid).unsqueeze(0),
            'render_images': torch.stack(images), 'render_masks': torch.stack(masks),
            'render_c2ws': torch.stack(c2ws), 'render_intrs': torch.stack(intrs),
            'source_c2ws': self._c2w(source_r, source_t).unsqueeze(0),
            'source_intrs': source_intr.unsqueeze(0), 'render_bg_colors': torch.ones(count, 3),
            'smplx_params': smplx_params, 'uid': uid, 'render_view_ids': target_views,
        }
