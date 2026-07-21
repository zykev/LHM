# -*- coding: utf-8 -*-
"""
Dress4DLHMDataset — 4DDress 多视角数据集适配 LHM 训练格式。

依赖 prepare_4ddress.py 的输出：
  {root}/{uid}/smplx_params/{frame:05d}.json   — 前视角相机坐标系 SMPL-X
  {root}/{uid}/cameras.json                    — 所有相机内外参
  {root}/{uid}/flame_params/{frame:05d}.json   — FLAME bbox（可选，用于 head crop）

原始图像从 raw_data_dir 读取：
  {raw_data_dir}/{subject}/{outfit}/{take}/Capture/{cam_id}/images/*.png
  {raw_data_dir}/{subject}/{outfit}/{take}/Capture/{cam_id}/masks/*.png
"""

import glob
import json
import os
import pickle
import random

import cv2
import numpy as np
import torch

from .base import BaseDataset

CAMERA_VIEWS = ['0004', '0028', '0052', '0076']
ASPECT_HW = 5.0 / 3.0


# ---------------------------------------------------------------------------
# 图像/内参工具
# ---------------------------------------------------------------------------

def _find_capture_file(cam_raw_dir: str, frame_idx: int, subdir: str) -> str:
    """构造 4DDress Capture/{cam}/{subdir} 下的文件路径。
    命名规则：images/capture-f{frame:05d}.png，masks/mask-f{frame:05d}.png。
    """
    prefix = 'capture' if subdir == 'images' else 'mask'
    path = os.path.join(cam_raw_dir, subdir, f'{prefix}-f{frame_idx:05d}.png')
    return path if os.path.exists(path) else ''


# ---------------------------------------------------------------------------
# Match LHM's original preprocessing: resize the long edge, crop around the
# optical axis using the foreground mask, then resize to the network size.
# The crop is centered on (cx, cy), which is essential because the legacy DGR
# rasterizer uses a symmetric FoV projection and does not consume cx/cy.
# ---------------------------------------------------------------------------

MAX_TGT_SIZE = 896


def _resize_keepaspect(img: np.ndarray, max_tgt_size: int):
    h, w = img.shape[:2]
    ratio = max_tgt_size / max(h, w)
    new_h, new_w = round(h * ratio), round(w * ratio)
    interp = cv2.INTER_AREA if ratio < 1 else cv2.INTER_LINEAR
    return cv2.resize(img, (new_w, new_h), interpolation=interp), ratio


def _center_crop_by_mask(img: np.ndarray, mask: np.ndarray, cx: float, cy: float,
                         aspect_standard: float, enlarge_ratio=(1.0, 1.0)):
    """Crop a symmetric rectangle around the calibrated optical axis."""
    ys, xs = np.where(mask > 0)
    height, width = img.shape[:2]
    if len(xs) == 0 or len(ys) == 0:
        return img, mask, 0, 0

    x_min, x_max, y_min, y_max = xs.min(), xs.max(), ys.min(), ys.max()
    cx_i, cy_i = int(round(cx)), int(round(cy))
    half_w = max(abs(cx_i - x_min), abs(cx_i - x_max))
    half_h = max(abs(cy_i - y_min), abs(cy_i - y_max))
    half_w_raw, half_h_raw = half_w, half_h
    if half_h / max(half_w, 1) >= aspect_standard:
        half_w = round(half_h / aspect_standard)
    else:
        half_h = round(half_w * aspect_standard)

    max_half_w = min(cx_i, width - cx_i)
    max_half_h = min(cy_i, height - cy_i)
    if half_h > max_half_h:
        half_w, half_h = round(half_h_raw / aspect_standard), half_h_raw
    if half_w > max_half_w:
        half_h, half_w = round(half_w_raw * aspect_standard), half_w_raw

    if abs(enlarge_ratio[0] - 1) > 0.01 or abs(enlarge_ratio[1] - 1) > 0.01:
        min_ratio, max_ratio = enlarge_ratio
        max_ratio = min(max_ratio, max_half_h / max(half_h, 1), max_half_w / max(half_w, 1))
        min_ratio = min(min_ratio, max_ratio)
        ratio = np.random.rand() * (max_ratio - min_ratio) + min_ratio
        half_h, half_w = round(ratio * half_h), round(ratio * half_w)

    half_h, half_w = min(half_h, max_half_h), min(half_w, max_half_w)
    offset_x, offset_y = cx_i - half_w, cy_i - half_h
    return (
        img[offset_y:offset_y + 2 * half_h, offset_x:offset_x + 2 * half_w],
        mask[offset_y:offset_y + 2 * half_h, offset_x:offset_x + 2 * half_w],
        offset_x,
        offset_y,
    )


def _calc_tgt_hw(aspect_standard: float, target_w: int, multiply: int):
    target_h = max(int((target_w * aspect_standard) / multiply) * multiply, multiply)
    target_w = max(int(target_w / multiply) * multiply, multiply)
    return target_h, target_w


def _load_view(img_path: str, mask_path: str, K_raw: np.ndarray, target_w: int,
               multiply: int, aspect_standard: float = ASPECT_HW,
               max_tgt_size: int = MAX_TGT_SIZE, enlarge_ratio=(1.0, 1.0),
               infer_white_background: bool = False):
    """Apply LHM preprocessing and propagate every image-space K transform."""
    target_h, target_w = _calc_tgt_hw(aspect_standard, target_w, multiply)

    img_bgr = cv2.imread(img_path) if img_path else None
    if img_bgr is None:
        img_t = torch.ones(3, target_h, target_w, dtype=torch.float32)
        mask_t = torch.zeros(1, target_h, target_w, dtype=torch.float32)
        intr = torch.eye(4, dtype=torch.float32)
        intr[0, 2], intr[1, 2] = target_w / 2, target_h / 2
        return img_t, mask_t, intr

    mask_gray = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path else None
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if mask_gray is not None:
        mask = (mask_gray > 127).astype(np.float32)
    elif infer_white_background:
        # HuGe100K has no separate alpha/mask; this is its documented
        # foreground definition and must also drive the calibrated crop.
        mask = (~np.all(img_bgr >= 245, axis=-1)).astype(np.float32)
    else:
        mask = np.ones(img.shape[:2], dtype=np.float32)

    # 背景合成纯白
    img = img * mask[:, :, None] + 1.0 * (1 - mask[:, :, None])

    K = K_raw.astype(np.float64).copy()
    img, ratio0 = _resize_keepaspect(img, max_tgt_size)
    mask, _ = _resize_keepaspect(mask, max_tgt_size)
    K[0, 0] *= ratio0; K[0, 2] *= ratio0
    K[1, 1] *= ratio0; K[1, 2] *= ratio0

    img, mask, offset_x, offset_y = _center_crop_by_mask(
        img, mask, K[0, 2], K[1, 2], aspect_standard, enlarge_ratio
    )
    K[0, 2] -= offset_x
    K[1, 2] -= offset_y

    crop_h, crop_w = img.shape[:2]
    ratio_y, ratio_x = target_h / crop_h, target_w / crop_w
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    K[0, 0] *= ratio_x; K[0, 2] *= ratio_x
    K[1, 1] *= ratio_y; K[1, 2] *= ratio_y

    # DGR's FoV projection has no principal-point term.  Do not hide an
    # off-centre calibration by overwriting K: reject it so the mismatch is
    # visible instead of silently producing shifted Gaussian renders.
    expected_center = np.array([target_w * 0.5, target_h * 0.5])
    actual_center = K[:2, 2]
    if not np.allclose(actual_center, expected_center, rtol=0.0, atol=1e-4):
        raise ValueError(
            'LHM crop did not center the optical axis: '
            f'got {actual_center.tolist()}, expected {expected_center.tolist()}'
        )

    intr = torch.eye(4, dtype=torch.float32)
    intr[:3, :3] = torch.from_numpy(K.astype(np.float32))

    img_t  = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1)
    mask_t = torch.from_numpy((mask > 0.5).astype(np.float32)).unsqueeze(0)
    return img_t, mask_t, intr


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LegacyDress4DLHMDataset(BaseDataset):
    """
    4DDress 数据集适配 LHM 训练。每个样本为一帧多视角数据：
      source: 前视角图像 + head crop
      target: 其余 N_tgt 视角图像 + mask（渲染监督）
      SMPL-X 与所有相机均保留在 4D-Dress 提供的世界坐标系中。

    Args:
        root_dirs:         已准备数据的根目录列表（prepare_4ddress.py 输出）
        meta_path:         train_list.json / val_list.json 路径
        raw_data_dir:      4DDress 原始数据路径（读取多视角图像用）
        source_image_res:  source 图像宽度，默认 512
        render_image:      {"low": 384} 渲染目标宽度
        sample_side_views: 目标视角数，最多 3
        use_flame:         是否加载 FLAME 参数
        src_head_size:     head crop 边长，默认 112
        multiply:          输出尺寸的 patch 对齐倍数
        max_tgt_size:      crop 前的最长边尺寸
        enlarge_ratio:     crop 随机放大范围
    """

    def __init__(
        self,
        root_dirs,
        meta_path: str,
        raw_data_dir: str = '',
        source_image_res: int = 512,
        render_image: dict = None,
        sample_side_views: int = 3,
        use_flame: bool = False,
        src_head_size: int = 112,
        multiply: int = 16,
        max_tgt_size: int = MAX_TGT_SIZE,
        enlarge_ratio=(1.0, 1.0),
        **kwargs,
    ):
        super().__init__(root_dirs, meta_path)
        self.root_dirs        = [root_dirs] if isinstance(root_dirs, str) else list(root_dirs)
        self.raw_data_dir     = raw_data_dir
        self.source_image_res = source_image_res
        self.render_image_res = (render_image or {}).get('high', (render_image or {}).get('low', 384))
        self.sample_side_views = sample_side_views
        self.multiply = multiply
        self.max_tgt_size = max_tgt_size
        self.enlarge_ratio = tuple(enlarge_ratio)
        self.use_flame        = use_flame
        self.src_head_size    = src_head_size

    def _find_prepared_dir(self, uid: str) -> str:
        for rd in self.root_dirs:
            d = os.path.join(rd, uid)
            if os.path.isdir(d):
                return d
        raise FileNotFoundError(f'找不到 uid={uid} 的已处理目录')

    def _raw_item_dir(self, uid: str) -> str:
        """uid = 'subject_outfit_take' → {raw_data_dir}/{subject}/{outfit}/{take}"""
        parts = uid.rsplit('_', 2)
        if len(parts) != 3:
            raise ValueError(f'无法解析 uid: {uid}')
        subject, outfit, take = parts
        return os.path.join(self.raw_data_dir, subject, outfit, take)

    def _crop_head(self, img_path: str, prepared_dir: str, frame_idx: int) -> torch.Tensor:
        """从 FLAME bbox 裁剪头部，返回 [3, head_size, head_size]；无 bbox 时返回零张量。"""
        hs = self.src_head_size
        flame_path = os.path.join(prepared_dir, 'flame_params', f'{frame_idx:05d}.json')
        if not os.path.exists(flame_path):
            return torch.zeros(3, hs, hs, dtype=torch.float32)
        try:
            with open(flame_path) as f:
                x1, y1, x2, y2 = [int(v) for v in json.load(f)['bbox']]
            img = cv2.imread(img_path)
            if img is None:
                return torch.zeros(3, hs, hs, dtype=torch.float32)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W = img.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
            if x2 <= x1 or y2 <= y1:
                return torch.zeros(3, hs, hs, dtype=torch.float32)
            head = cv2.resize(img[y1:y2, x1:x2], (hs, hs), interpolation=cv2.INTER_AREA)
            return torch.from_numpy(head.astype(np.float32) / 255.0).permute(2, 0, 1)
        except Exception:
            return torch.zeros(3, hs, hs, dtype=torch.float32)

    def inner_get_item(self, idx: int) -> dict:
        uid = self.uids[idx]
        prepared_dir = self._find_prepared_dir(uid)

        # --- cameras.json ---
        with open(os.path.join(prepared_dir, 'cameras.json')) as f:
            cameras = json.load(f)
        sorted_cams = sorted(cameras.items(), key=lambda x: x[1]['sorted_idx'])
        src_cam_id  = sorted_cams[0][0]
        # 所有 4 个相机均作为 target（含前视角自身，提供自重建监督）
        tgt_cam_ids = [cid for cid, _ in sorted_cams]
        src_cam     = cameras[src_cam_id]
        N_tgt       = len(tgt_cam_ids)

        # --- 随机帧 ---
        json_files = sorted(glob.glob(os.path.join(prepared_dir, 'smplx_params', '*.json')))
        if not json_files:
            raise FileNotFoundError(f'smplx_params 为空: {prepared_dir}')
        chosen = random.choice(json_files)
        frame_idx = int(os.path.splitext(os.path.basename(chosen))[0])

        # --- SMPL-X 参数 ---
        with open(chosen) as f:
            sp = json.load(f)

        def _t(key, shape):
            return torch.tensor(sp[key], dtype=torch.float32).reshape(shape)

        betas     = _t('betas',     (10,))
        root_pose = _t('root_pose', (3,))
        body_pose = _t('body_pose', (21, 3))
        jaw_pose  = _t('jaw_pose',  (3,))
        leye_pose = _t('leye_pose', (3,))
        reye_pose = _t('reye_pose', (3,))
        lhand     = _t('lhand_pose', (15, 3))
        rhand     = _t('rhand_pose', (15, 3))
        expr      = _t('expr',       (100,))
        trans     = _t('trans',      (3,))

        def _expand(t):
            return t.unsqueeze(0).expand(N_tgt, *t.shape).clone()

        smplx_params = {
            'root_pose':  _expand(root_pose),
            'body_pose':  _expand(body_pose),
            'jaw_pose':   _expand(jaw_pose),
            'leye_pose':  _expand(leye_pose),
            'reye_pose':  _expand(reye_pose),
            'lhand_pose': _expand(lhand),
            'rhand_pose': _expand(rhand),
            'expr':       _expand(expr),
            'trans':      _expand(trans),
            'betas':      betas,
        }

        # --- 相机外参（4D-Dress world c2w）---
        # SMPL-X pkl、R/T 和所有 view 均已在同一世界坐标系中验证对齐。这里不能
        # 以 source view 为原点重写 target camera；renderer 会自行将 c2w 求逆为 w2c。
        c2w_src = np.array(src_cam['c2w'], dtype=np.float64)
        render_c2ws = torch.stack([
            torch.tensor(np.array(cameras[cid]['c2w'], dtype=np.float64), dtype=torch.float32)
            for cid in tgt_cam_ids
        ])  # [N_tgt, 4, 4]
        source_c2ws = torch.tensor(c2w_src, dtype=torch.float32).unsqueeze(0)

        render_bg_colors = torch.ones(N_tgt, 3)

        # --- 图像 + mask + 内参：LHM 原始 crop/resize，K 同步更新。---
        if self.raw_data_dir:
            raw_item = self._raw_item_dir(uid)
            src_cam_raw  = os.path.join(raw_item, 'Capture', src_cam_id)
            src_img_path  = _find_capture_file(src_cam_raw, frame_idx, 'images')
            src_mask_path = _find_capture_file(src_cam_raw, frame_idx, 'masks')
        else:
            src_img_path  = os.path.join(prepared_dir, 'imgs_png', f'{frame_idx:05d}.png')
            src_mask_path = ''

        src_image, _src_mask, source_intr = _load_view(
            src_img_path, src_mask_path, np.array(src_cam['intrinsics']),
            self.source_image_res, self.multiply,
            max_tgt_size=self.max_tgt_size, enlarge_ratio=self.enlarge_ratio,
        )
        src_image    = src_image.unsqueeze(0)     # [1, 3, H_src, W_src]
        source_intrs = source_intr.unsqueeze(0)   # [1, 4, 4]

        render_images, render_masks, render_intrs = [], [], []
        for cid in tgt_cam_ids:
            if self.raw_data_dir:
                tgt_cam_raw   = os.path.join(raw_item, 'Capture', cid)
                tgt_img_path  = _find_capture_file(tgt_cam_raw, frame_idx, 'images')
                tgt_mask_path = _find_capture_file(tgt_cam_raw, frame_idx, 'masks')
            else:
                tgt_img_path, tgt_mask_path = '', ''
            img, mask, intr = _load_view(
                tgt_img_path, tgt_mask_path, np.array(cameras[cid]['intrinsics']),
                self.render_image_res, self.multiply,
                max_tgt_size=self.max_tgt_size, enlarge_ratio=self.enlarge_ratio,
            )
            render_images.append(img)
            render_masks.append(mask)
            render_intrs.append(intr)

        render_images = torch.stack(render_images)  # [N_tgt, 3, H, W]
        render_masks  = torch.stack(render_masks)   # [N_tgt, 1, H, W]
        render_intrs  = torch.stack(render_intrs)   # [N_tgt, 4, 4]

        src_head_rgb = self._crop_head(src_img_path, prepared_dir, frame_idx).unsqueeze(0)

        flame_params = None
        if self.use_flame:
            fp = os.path.join(prepared_dir, 'flame_params', f'{frame_idx:05d}.json')
            if os.path.exists(fp):
                with open(fp) as f:
                    flame_params = json.load(f)

        sample = {
            'src_images':       src_image,        # [1, 3, H_src, W_src]
            'source_head_rgbs': src_head_rgb,     # [1, 3, hs, hs]
            'render_images':    render_images,    # [N_tgt, 3, H, W]
            'render_masks':     render_masks,     # [N_tgt, 1, H, W]
            'render_c2ws':      render_c2ws,      # [N_tgt, 4, 4]
            'render_intrs':     render_intrs,     # [N_tgt, 4, 4]
            'source_c2ws':      source_c2ws,      # [1, 4, 4]
            'source_intrs':     source_intrs,     # [1, 4, 4]
            'render_bg_colors': render_bg_colors, # [N_tgt, 3]
            'smplx_params':     smplx_params,
            'uid':              uid,
            'frame_idx':        frame_idx,
        }
        if flame_params is not None:
            sample['flame_params'] = flame_params
        return sample


class Dress4DLHMDataset(BaseDataset):
    """LHM-Track 4D-Dress adapter.

    ``root_dirs`` holds only Track metadata; images, cameras and SMPL-X stay
    in ``raw_data_dir``.  Training consumes one ``#gNN`` target group, while
    evaluation de-duplicates the groups and renders all 24 views.
    """

    def __init__(self, root_dirs, meta_path, raw_data_dir='', source_image_res=512,
                 render_image=None, multiply=16, max_tgt_size=MAX_TGT_SIZE,
                 enlarge_ratio=(1., 1.), eval_all_views=False, src_head_size=112,
                 **kwargs):
        super().__init__(root_dirs, meta_path)
        self.root_dirs = [root_dirs] if isinstance(root_dirs, str) else list(root_dirs)
        self.raw_data_dir = raw_data_dir
        self.source_image_res = source_image_res
        self.render_image_res = (render_image or {}).get('high', (render_image or {}).get('low', 384))
        self.multiply, self.max_tgt_size = multiply, max_tgt_size
        self.enlarge_ratio = tuple(enlarge_ratio)
        self.eval_all_views, self.src_head_size = eval_all_views, src_head_size
        with open(os.path.join(self.root_dirs[0], 'label', 'dataset_meta.json')) as f:
            meta = json.load(f)
        self.source_view = meta['source_views'][0]
        self.target_groups = meta['target_view_groups']
        self.all_views = [f'{index:02d}' for index in range(meta['view_count'])]
        selected = kwargs.get('eval_sample_ids')
        if selected is not None:
            selected = [self._canonical_selected_uid(uid) for uid in selected]
            # Eval normally uses val_list.json.  An explicit CLI selection is
            # intentionally allowed to address a prepared train-list sample as
            # well (useful for inspecting one item without changing the split).
            entries = list(self.uids)
            available = {entry.split('#', 1)[0] for entry in entries}
            missing = [uid for uid in selected if uid not in available]
            if missing:
                for root in self.root_dirs:
                    path = os.path.join(root, 'label', 'train_list.json')
                    if os.path.isfile(path):
                        with open(path) as handle:
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
            self.uids = list(dict.fromkeys(uid.split('#', 1)[0] for uid in self.uids))

    @staticmethod
    def _parse_entry(entry):
        uid, sep, group = entry.partition('#')
        return uid, int(group[1:]) if sep else 0

    @staticmethod
    def _canonical_selected_uid(uid):
        """Accept the same bare 4D-Dress ID form as Track prepare."""
        uid = str(uid).strip().replace('\\', '/')
        if '/' in uid:
            return uid
        fields = uid.split('_')
        if len(fields) < 4:
            raise ValueError(
                'bare 4D-Dress sample id must be '
                '<subject>_<outfit>_<take>_<frame>: ' + uid
            )
        return f'{"_".join(fields[:2])}/{uid}'

    def _face_bbox(self, uid):
        for root in self.root_dirs:
            path = os.path.join(root, 'face_bbox', f'{uid}.json')
            if os.path.isfile(path):
                with open(path) as f:
                    return json.load(f)['bbox']
        return None

    def _head_crop(self, source_path, uid):
        bbox = self._face_bbox(uid)
        if bbox is None:
            return torch.zeros(3, self.src_head_size, self.src_head_size)
        image = cv2.imread(source_path)
        if image is None:
            raise FileNotFoundError(source_path)
        x1, y1, x2, y2 = map(int, bbox)
        h, w = image.shape[:2]
        x1, x2, y1, y2 = max(0, x1), min(w, x2), max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f'invalid face bbox for {uid}: {bbox}')
        crop = cv2.resize(image[y1:y2, x1:x2], (self.src_head_size, self.src_head_size), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.).permute(2, 0, 1)

    @staticmethod
    def _c2w(camera):
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = np.asarray(camera['R'], dtype=np.float32)
        w2c[:3, 3] = np.asarray(camera['T'], dtype=np.float32)
        return torch.from_numpy(np.linalg.inv(w2c).astype(np.float32))

    @staticmethod
    def _array(data, key, shape):
        return torch.as_tensor(data[key], dtype=torch.float32).reshape(shape)

    def inner_get_item(self, idx):
        entry = self.uids[idx]
        uid, group_index = self._parse_entry(entry)
        sample_dir = os.path.join(self.raw_data_dir, uid)
        with open(os.path.join(sample_dir, 'camera.json')) as f:
            cameras = json.load(f)
        with open(next(iter(sorted(glob.glob(os.path.join(sample_dir, 'smplx', '*.pkl'))))), 'rb') as f:
            params = pickle.load(f)
        target_views = self.all_views if self.eval_all_views else self.target_groups[group_index]
        source_path = os.path.join(sample_dir, 'image', f'{self.source_view}.png')
        source_mask = os.path.join(sample_dir, 'mask', f'{self.source_view}.png')
        source_image, _, source_intr = _load_view(source_path, source_mask, np.asarray(cameras[self.source_view]['K']),
                                                    self.source_image_res, self.multiply, max_tgt_size=self.max_tgt_size,
                                                    enlarge_ratio=self.enlarge_ratio)
        images, masks, intrs, c2ws = [], [], [], []
        for view in target_views:
            image, mask, intr = _load_view(os.path.join(sample_dir, 'image', f'{view}.png'),
                                           os.path.join(sample_dir, 'mask', f'{view}.png'),
                                           np.asarray(cameras[view]['K']), self.render_image_res, self.multiply,
                                           max_tgt_size=self.max_tgt_size, enlarge_ratio=self.enlarge_ratio)
            images.append(image); masks.append(mask); intrs.append(intr); c2ws.append(self._c2w(cameras[view]))
        nv = len(target_views)
        expand = lambda value: value.unsqueeze(0).expand(nv, *value.shape).clone()
        smplx = {
            'root_pose': expand(self._array(params, 'global_orient', (3,))),
            'body_pose': expand(self._array(params, 'body_pose', (21, 3))),
            'jaw_pose': expand(self._array(params, 'jaw_pose', (3,))),
            'leye_pose': expand(self._array(params, 'leye_pose', (3,))),
            'reye_pose': expand(self._array(params, 'reye_pose', (3,))),
            'lhand_pose': expand(self._array(params, 'left_hand_pose', (12,))),
            'rhand_pose': expand(self._array(params, 'right_hand_pose', (12,))),
            'expr': expand(self._array(params, 'expression', (100,))),
            'trans': expand(self._array(params, 'transl', (3,))),
            'betas': self._array(params, 'betas', (10,)),
        }
        return {
            'src_images': source_image.unsqueeze(0),
            'source_head_rgbs': self._head_crop(source_path, uid).unsqueeze(0),
            'render_images': torch.stack(images), 'render_masks': torch.stack(masks),
            'render_c2ws': torch.stack(c2ws), 'render_intrs': torch.stack(intrs),
            'source_c2ws': self._c2w(cameras[self.source_view]).unsqueeze(0),
            'source_intrs': source_intr.unsqueeze(0), 'render_bg_colors': torch.ones(nv, 3),
            'smplx_params': smplx, 'uid': uid, 'render_view_ids': target_views,
        }
