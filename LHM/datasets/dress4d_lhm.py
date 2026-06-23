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
import random

import cv2
import numpy as np
import torch

from .base import BaseDataset

CAMERA_VIEWS = ['0004', '0028', '0052', '0076']
ASPECT_HW = 5.0 / 3.0  # LHM 期望的高:宽


# ---------------------------------------------------------------------------
# 图像/内参工具
# ---------------------------------------------------------------------------

def _load_image(path: str, target_w: int,
                crop_dx: int, crop_dy: int, crop_w: int, crop_h: int) -> torch.Tensor:
    """读取 BGR 图像，center-crop，resize 到 (target_w, target_w*ASPECT_HW)，返回 [3,H,W] float32。"""
    target_h = int(target_w * ASPECT_HW)
    img = cv2.imread(path)
    if img is None:
        return torch.zeros(3, target_h, target_w, dtype=torch.float32)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)[crop_dy:crop_dy + crop_h, crop_dx:crop_dx + crop_w]
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
    return torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)


def _load_mask(path: str, target_w: int,
               crop_dx: int, crop_dy: int, crop_w: int, crop_h: int) -> torch.Tensor:
    """读取 mask，返回 [1,H,W] float32；文件不存在时返回全 1（全前景）。"""
    target_h = int(target_w * ASPECT_HW)
    if not os.path.exists(path):
        return torch.ones(1, target_h, target_w, dtype=torch.float32)
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return torch.ones(1, target_h, target_w, dtype=torch.float32)
    mask = mask[crop_dy:crop_dy + crop_h, crop_dx:crop_dx + crop_w]
    mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0)


def _crop_params(cam_info: dict) -> tuple:
    """由 cameras.json 相机信息计算 5:3 center-crop 参数，返回 (W_crop, H_crop, dx, dy)。"""
    W, H = cam_info['img_wh']
    if H / W > ASPECT_HW:
        H_crop = int(W * ASPECT_HW)
        return W, H_crop, 0, (H - H_crop) // 2
    elif H / W < ASPECT_HW:
        W_crop = int(H / ASPECT_HW)
        return W_crop, H, (W - W_crop) // 2, 0
    return W, H, 0, 0


def _build_intrinsic_4x4(cam_info: dict, target_w: int) -> torch.Tensor:
    """构建 crop+resize 后的 4×4 像素空间内参矩阵。

    LHM 的渲染器约定主点严格位于图像中心（参见 runners/infer/utils.py 中
    intr[0,2]=W//2, intr[1,2]=H//2 的写法，渲染分辨率也由 princpt*2 反推），
    而非使用真实标定的 cx,cy（标定主点几乎不会恰好居中，会导致渲染分辨率
    与数据集实际分辨率不一致）。因此这里强制把主点设为目标分辨率中心。
    """
    K = np.array(cam_info['intrinsics'])
    W_crop, H_crop, dx, dy = _crop_params(cam_info)
    target_h = int(target_w * ASPECT_HW)
    sx, sy = target_w / W_crop, target_h / H_crop
    mat = torch.eye(4, dtype=torch.float32)
    mat[0, 0] = K[0, 0] * sx
    mat[1, 1] = K[1, 1] * sy
    mat[0, 2] = target_w / 2
    mat[1, 2] = target_h / 2
    return mat


def _find_capture_file(cam_raw_dir: str, frame_idx: int, subdir: str) -> str:
    """构造 4DDress Capture/{cam}/{subdir} 下的文件路径。
    命名规则：images/capture-f{frame:05d}.png，masks/mask-f{frame:05d}.png。
    """
    prefix = 'capture' if subdir == 'images' else 'mask'
    path = os.path.join(cam_raw_dir, subdir, f'{prefix}-f{frame_idx:05d}.png')
    return path if os.path.exists(path) else ''


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class Dress4DLHMDataset(BaseDataset):
    """
    4DDress 数据集适配 LHM 训练。每个样本为一帧多视角数据：
      source: 前视角图像 + head crop
      target: 其余 N_tgt 视角图像 + mask（渲染监督）
      SMPL-X 在前视角相机坐标系下，target c2w 为相对 source 的变换。

    Args:
        root_dirs:         已准备数据的根目录列表（prepare_4ddress.py 输出）
        meta_path:         train_list.json / val_list.json 路径
        raw_data_dir:      4DDress 原始数据路径（读取多视角图像用）
        source_image_res:  source 图像宽度，默认 512
        render_image:      {"low": 384} 渲染目标宽度
        sample_side_views: 目标视角数，最多 3
        use_flame:         是否加载 FLAME 参数
        src_head_size:     head crop 边长，默认 112
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
        **kwargs,
    ):
        super().__init__(root_dirs, meta_path)
        self.root_dirs        = [root_dirs] if isinstance(root_dirs, str) else list(root_dirs)
        self.raw_data_dir     = raw_data_dir
        self.source_image_res = source_image_res
        self.render_image_res = (render_image or {}).get('low', 384)
        self.sample_side_views = sample_side_views
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

        # --- 相机：相对 c2w + 内参 ---
        c2w_src = np.array(src_cam['c2w'], dtype=np.float64)
        w2c_src = np.linalg.inv(c2w_src)

        render_c2ws, render_intrs = [], []
        for cid in tgt_cam_ids:
            c2w_tgt = np.array(cameras[cid]['c2w'], dtype=np.float64)
            render_c2ws.append(torch.tensor(w2c_src @ c2w_tgt, dtype=torch.float32))
            render_intrs.append(_build_intrinsic_4x4(cameras[cid], self.render_image_res))

        render_c2ws  = torch.stack(render_c2ws)   # [N_tgt, 4, 4]
        render_intrs = torch.stack(render_intrs)  # [N_tgt, 4, 4]
        source_c2ws  = torch.eye(4).unsqueeze(0)  # [1, 4, 4]
        source_intrs = _build_intrinsic_4x4(src_cam, self.source_image_res).unsqueeze(0)

        render_bg_colors = torch.ones(N_tgt, 3)

        # --- 图像 ---
        src_cp = _crop_params(src_cam)  # (W_crop, H_crop, dx, dy)

        if self.raw_data_dir:
            raw_item = self._raw_item_dir(uid)
            src_cam_raw = os.path.join(raw_item, 'Capture', src_cam_id)
            src_img_path = _find_capture_file(src_cam_raw, frame_idx, 'images')
        else:
            src_img_path = os.path.join(prepared_dir, 'imgs_png', f'{frame_idx:05d}.png')

        src_image = _load_image(
            src_img_path, self.source_image_res, src_cp[2], src_cp[3], src_cp[0], src_cp[1]
        ).unsqueeze(0)  # [1, 3, H_src, W_src]

        render_images, render_masks = [], []
        for cid in tgt_cam_ids:
            tgt_cp = _crop_params(cameras[cid])
            if self.raw_data_dir:
                tgt_cam_raw  = os.path.join(raw_item, 'Capture', cid)
                tgt_img_path  = _find_capture_file(tgt_cam_raw, frame_idx, 'images')
                tgt_mask_path = _find_capture_file(tgt_cam_raw, frame_idx, 'masks')
                render_images.append(_load_image(
                    tgt_img_path, self.render_image_res, tgt_cp[2], tgt_cp[3], tgt_cp[0], tgt_cp[1]))
                render_masks.append(_load_mask(
                    tgt_mask_path, self.render_image_res, tgt_cp[2], tgt_cp[3], tgt_cp[0], tgt_cp[1]))
            else:
                H_r = int(self.render_image_res * ASPECT_HW)
                render_images.append(torch.zeros(3, H_r, self.render_image_res))
                render_masks.append(torch.ones(1, H_r, self.render_image_res))

        render_images = torch.stack(render_images)  # [N_tgt, 3, H, W]
        render_masks  = torch.stack(render_masks)   # [N_tgt, 1, H, W]

        # GS3DRenderer 渲染时背景统一合成成纯白（render_bg_colors=ones），但这里
        # render_images 加载的是真实拍摄照片，背景是真实场景，跟渲染背景天然不一致。
        # 用 mask 把 GT 背景也替换成纯白，和渲染器约定保持一致。
        render_images = render_images * render_masks + (1.0 - render_masks)

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
