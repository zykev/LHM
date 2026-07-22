"""4D-Dress quantitative evaluation through the official LHM inference path."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import cv2
import kornia.metrics as kornia_metrics
import numpy as np
import torch
from PIL import Image

from LHM.datasets.dress4d_lhm import _load_view
from LHM.losses import LPIPSLoss


SCHEMA = "4ddress_static_camera_sequence_v1"


def _canonical_uid(value: str) -> str:
    value = str(value).strip().replace("\\", "/")
    if "/" in value:
        return value
    fields = value.split("_")
    if len(fields) < 4:
        raise ValueError(f"invalid 4D-Dress sample id: {value}")
    return f"{'_'.join(fields[:2])}/{value}"


def _read_list(path: str) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        values = [line.split("#", 1)[0].strip() for line in handle]
    values = [_canonical_uid(value) for value in values if value]
    if not values:
        raise ValueError(f"sample list is empty: {path}")
    return list(dict.fromkeys(values))


def _selected_uids(cfg, metadata_root: Path) -> list[str]:
    sample_ids = cfg.get("sample_ids")
    sample_list = cfg.get("sample_list")
    if sample_ids and sample_list:
        raise ValueError("use --sample-id or --sample-list, not both")
    if sample_ids:
        return list(dict.fromkeys(_canonical_uid(value) for value in sample_ids))
    if sample_list:
        return _read_list(sample_list)
    sequence_root = metadata_root / "motion_sequences"
    return sorted(path.parent.relative_to(sequence_root).as_posix()
                  for path in sequence_root.glob("*/*/sequence.json"))


def _c2w(frame: dict) -> torch.Tensor:
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, :3] = np.asarray(frame["R"], dtype=np.float32)
    w2c[:3, 3] = np.asarray(frame["T"], dtype=np.float32)
    return torch.from_numpy(np.linalg.inv(w2c).astype(np.float32))


def _head_crop(source_path: Path, metadata_root: Path, uid: str, size: int) -> torch.Tensor:
    bbox_path = metadata_root / "face_bbox" / Path(uid).with_suffix(".json")
    if not bbox_path.is_file():
        raise FileNotFoundError(f"missing Track face bbox: {bbox_path}")
    with bbox_path.open(encoding="utf-8") as handle:
        x1, y1, x2, y2 = map(int, json.load(handle)["bbox"])
    image = cv2.imread(str(source_path))
    if image is None:
        raise FileNotFoundError(source_path)
    h, w = image.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid face bbox for {uid}: {(x1, y1, x2, y2)}")
    crop = cv2.resize(image[y1:y2, x1:x2], (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(rgb).permute(2, 0, 1)


def _array(data: dict, key: str, shape: tuple[int, ...]) -> torch.Tensor:
    return torch.as_tensor(data[key], dtype=torch.float32).reshape(shape)


def _pad_expression(value) -> torch.Tensor:
    expression = torch.as_tensor(value, dtype=torch.float32).reshape(-1)
    if expression.numel() > 100:
        raise ValueError(f"expression has {expression.numel()} values; expected <= 100")
    return torch.nn.functional.pad(expression, (0, 100 - expression.numel()))


def _smplx_params(path: Path, views: int) -> dict[str, torch.Tensor]:
    with path.open("rb") as handle:
        raw = pickle.load(handle)
    expand = lambda value: value.unsqueeze(0).unsqueeze(0).expand(1, views, *value.shape).clone()
    return {
        "root_pose": expand(_array(raw, "global_orient", (3,))),
        "body_pose": expand(_array(raw, "body_pose", (21, 3))),
        "jaw_pose": expand(_array(raw, "jaw_pose", (3,))),
        "leye_pose": expand(_array(raw, "leye_pose", (3,))),
        "reye_pose": expand(_array(raw, "reye_pose", (3,))),
        "lhand_pose": expand(_array(raw, "left_hand_pose", (12,))),
        "rhand_pose": expand(_array(raw, "right_hand_pose", (12,))),
        "expr": expand(_pad_expression(raw["expression"])),
        "trans": expand(_array(raw, "transl", (3,))),
        "betas": _array(raw, "betas", (10,)).unsqueeze(0),
    }


def _metric_values(pred: torch.Tensor, gt: torch.Tensor, masks: torch.Tensor, lpips) -> dict[str, float]:
    psnr_values, ssim_values, lpips_values = [], [], []
    for index in range(pred.shape[0]):
        coords = torch.nonzero(masks[index, 0] > 0, as_tuple=False)
        if coords.numel():
            y0, x0 = coords.min(dim=0)[0]
            y1, x1 = coords.max(dim=0)[0]
            pred_crop = pred[index:index + 1, :, y0:y1 + 1, x0:x1 + 1]
            gt_crop = gt[index:index + 1, :, y0:y1 + 1, x0:x1 + 1]
        else:
            pred_crop, gt_crop = pred[index:index + 1], gt[index:index + 1]
        psnr_values.append(kornia_metrics.psnr(pred_crop, gt_crop, max_val=1.0))
        ssim_values.append(kornia_metrics.ssim(pred_crop, gt_crop, window_size=11, max_val=1.0).mean())
        lpips_values.append(lpips(pred_crop.unsqueeze(1), gt_crop.unsqueeze(1), is_training=False))
    return {
        "psnr": torch.stack(psnr_values).mean().item(),
        "ssim": torch.stack(ssim_values).mean().item(),
        "lpips": torch.stack(lpips_values).mean().item(),
    }


def _save_renders(root: Path, uid: str, view_ids: list[str], renders: torch.Tensor) -> None:
    output = root / Path(uid)
    output.mkdir(parents=True, exist_ok=True)
    for view_id, image in zip(view_ids, renders):
        array = (image.permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(array).save(output / f"{view_id}.png")


@torch.no_grad()
def evaluate_4ddress(inferrer) -> None:
    """Run official ``infer_single_view -> animation_infer`` for 4D-Dress."""
    cfg = inferrer.cfg
    metadata_root = Path(cfg.get("metadata_root", "../LHM_Track/train_data/4ddress_lhm"))
    dataset_root = Path(cfg.get("dataset_root", "../LHM_Track/.datasets/4d-dress"))
    output_root = Path(cfg.get("output_dir", "./outputs/4ddress_eval"))
    source_size = int(cfg.get("source_size", 512))
    render_size = int(cfg.get("render_size", 384))
    head_size = int(cfg.get("src_head_size", 112))
    aspect_hw = float(cfg.get("crop_aspect_hw", 1.0))
    multiply = int(cfg.get("multiply", 16))
    max_tgt_size = int(cfg.get("max_tgt_size", 896))

    if aspect_hw != 1.0:
        raise ValueError("4D-Dress official evaluation is defined for crop_aspect_hw=1.0")
    if not metadata_root.is_dir() or not dataset_root.is_dir():
        raise FileNotFoundError(f"metadata_root={metadata_root}, dataset_root={dataset_root}")

    device = inferrer.device
    # ``HumanLRMInferrer`` normally constructs this in ``__init__``.  Keep
    # the evaluator robust to older runners that initialise the official
    # inference object before the 4D-Dress mode is injected.
    if inferrer.model is None:
        inferrer.model = inferrer._build_model(cfg)
        if inferrer.model is None:
            raise RuntimeError("failed to construct the 4D-Dress inference model")
        inferrer.model.to(device)
    model = inferrer.model
    model.eval()
    model.to(dtype=torch.float32)
    perceptual = LPIPSLoss(device=device, prefech=False)
    per_sample = {}

    for uid in _selected_uids(cfg, metadata_root):
        manifest_path = metadata_root / "motion_sequences" / Path(uid) / "sequence.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"missing motion sequence manifest: {manifest_path}")
        with manifest_path.open(encoding="utf-8") as handle:
            sequence = json.load(handle)
        if sequence.get("schema") != SCHEMA:
            raise ValueError(f"unsupported motion sequence schema: {manifest_path}")
        if sequence.get("sample_id") != uid:
            raise ValueError(f"manifest sample_id mismatch: {manifest_path}")

        sample_dir = dataset_root / Path(uid)
        frames = sequence["frames"]
        source_view = sequence["source_view"]
        frame_by_id = {frame["view_id"]: frame for frame in frames}
        source_frame = frame_by_id[source_view]
        source_path = sample_dir / source_frame["image"]
        source_mask = sample_dir / source_frame["mask"]
        source_image, _, _ = _load_view(
            str(source_path), str(source_mask), np.asarray(source_frame["K"]), source_size,
            multiply, max_tgt_size=max_tgt_size, enlarge_ratio=(1.0, 1.0), aspect_standard=aspect_hw,
        )
        source_head = _head_crop(source_path, metadata_root, uid, head_size)

        gt_images, gt_masks, intrs, c2ws, view_ids = [], [], [], [], []
        for frame in frames:
            image, mask, intr = _load_view(
                str(sample_dir / frame["image"]), str(sample_dir / frame["mask"]),
                np.asarray(frame["K"]), render_size, multiply,
                max_tgt_size=max_tgt_size, enlarge_ratio=(1.0, 1.0), aspect_standard=aspect_hw,
            )
            gt_images.append(image)
            gt_masks.append(mask)
            intrs.append(intr)
            c2ws.append(_c2w(frame))
            view_ids.append(frame["view_id"])

        render_c2ws = torch.stack(c2ws).unsqueeze(0).to(device)
        render_intrs = torch.stack(intrs).unsqueeze(0).to(device)
        smplx = {key: value.to(device) for key, value in _smplx_params(sample_dir / sequence["smplx_pkl"], len(frames)).items()}
        backgrounds = torch.ones(1, len(frames), 3, device=device)

        gs_models, query_points, transform = model.infer_single_view(
            # Official inference expects [B, N_ref, C, H, W].  4D-Dress has
            # one fixed source view, hence both leading dimensions are one.
            source_image.unsqueeze(0).unsqueeze(0).to(device),
            source_head.unsqueeze(0).unsqueeze(0).to(device),
            None, None, render_c2ws, render_intrs, backgrounds, smplx,
        )
        smplx["transform_mat_neutral_pose"] = transform
        output = model.animation_infer(gs_models, query_points, smplx, render_c2ws, render_intrs, backgrounds)
        prediction = output["comp_rgb"].permute(0, 3, 1, 2).float().clamp(0, 1)
        gt = torch.stack(gt_images).to(device).float().clamp(0, 1)
        masks = torch.stack(gt_masks).to(device).float()
        metrics = _metric_values(prediction, gt, masks, perceptual)
        per_sample[uid] = metrics

        if cfg.get("save_render", False):
            _save_renders(output_root, uid, view_ids, prediction)
        print(f"{uid}: " + ", ".join(f"{key}={value:.4f}" for key, value in metrics.items()))

    if not per_sample:
        raise RuntimeError("no 4D-Dress samples selected")
    mean = {key: float(np.mean([metrics[key] for metrics in per_sample.values()]))
            for key in ("psnr", "ssim", "lpips")}
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"mean": mean, "samples": per_sample}, handle, indent=2)
    print("mean: " + ", ".join(f"{key}={value:.4f}" for key, value in mean.items()))
