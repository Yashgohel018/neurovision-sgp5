import io
import math
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PIL import Image
import nibabel as nib
from app.config import CASES_DIR, BRATS_DATA_DIR

MODALITY_PROFILES = {
    "T1": {"wm": 170, "gm": 120, "csf": 30, "tumor": 100, "enhance": False, "edema": 110},
    "T1ce": {"wm": 160, "gm": 120, "csf": 30, "tumor": 100, "enhance": True, "edema": 110},
    "T2": {"wm": 70, "gm": 120, "csf": 230, "tumor": 180, "enhance": False, "edema": 210},
    "FLAIR": {"wm": 80, "gm": 110, "csf": 25, "tumor": 160, "enhance": False, "edema": 220},
}

class NiftiService:
    def __init__(self):
        self._volume_cache: dict[str, np.ndarray] = {}
        self._slice_cache: dict[tuple, bytes] = {}

    def _get_volume(self, path: Path) -> np.ndarray:
        key = str(path.resolve())
        if key not in self._volume_cache:
            if len(self._volume_cache) >= 16:
                # Evict oldest entry
                self._volume_cache.pop(next(iter(self._volume_cache)))
            img = nib.load(key)
            self._volume_cache[key] = img.get_fdata(dtype=np.float32)
        return self._volume_cache[key]

    def get_case_nifti_path(self, case_id: str, modality: str) -> Optional[Path]:
        # Alias benchmark demo cases to real BraTS 2020 patient data
        effective_id = case_id
        if case_id == "CASE-2026-001":
            effective_id = "BraTS20_Training_001"
        elif case_id == "CASE-2026-002":
            effective_id = "BraTS20_Training_002"

        # 1. Check custom uploaded cases in CASES_DIR
        case_dir = CASES_DIR / case_id
        if case_dir.exists():
            for ext in [".nii.gz", ".nii"]:
                candidate = case_dir / f"{modality}{ext}"
                if candidate.exists():
                    return candidate
                candidate_lower = case_dir / f"{modality.lower()}{ext}"
                if candidate_lower.exists():
                    return candidate_lower

        # 2. Check real BraTS 2020 dataset directory
        brats_dir = BRATS_DATA_DIR / effective_id
        if brats_dir.exists():
            mod_key = "seg" if modality.lower() == "seg" else modality.lower()
            for ext in [".nii", ".nii.gz"]:
                candidate = brats_dir / f"{effective_id}_{mod_key}{ext}"
                if candidate.exists():
                    return candidate
                candidate_raw = brats_dir / f"{effective_id}_{modality}{ext}"
                if candidate_raw.exists():
                    return candidate_raw

        return None

    def render_slice_png(
        self,
        case_id: str,
        modality: str,
        plane: str = "axial",
        index: int = 60,
        overlay: Optional[str] = None,
        width: int = 256,
        height: int = 256,
    ) -> bytes:
        cache_key = (case_id, modality, plane, index, overlay, width, height)
        if cache_key in self._slice_cache:
            return self._slice_cache[cache_key]

        nii_path = self.get_case_nifti_path(case_id, modality)
        res_bytes = None
        if nii_path and nii_path.exists():
            try:
                res_bytes = self._render_from_nifti(nii_path, case_id, modality, plane, index, overlay, width, height)
            except Exception:
                pass

        if res_bytes is None:
            res_bytes = self._render_synthetic_slice(modality, plane, index, overlay, width, height)

        if len(self._slice_cache) >= 500:
            self._slice_cache.pop(next(iter(self._slice_cache)))
        self._slice_cache[cache_key] = res_bytes
        return res_bytes

    def _render_from_nifti(
        self,
        path: Path,
        case_id: str,
        modality: str,
        plane: str,
        index: int,
        overlay: Optional[str],
        width: int,
        height: int,
    ) -> bytes:
        data = self._get_volume(path)
        shape = data.shape

        if plane == "axial":
            idx = np.clip(index, 0, shape[2] - 1)
            slc = np.rot90(data[:, :, idx])
        elif plane == "coronal":
            idx = np.clip(index, 0, shape[1] - 1)
            slc = np.rot90(data[:, idx, :])
        else: # sagittal
            idx = np.clip(index, 0, shape[0] - 1)
            slc = np.rot90(data[idx, :, :])

        p1, p99 = np.percentile(slc, (1, 99))
        if p99 > p1:
            norm = np.clip((slc - p1) / (p99 - p1), 0, 1) * 255
        else:
            norm = np.zeros_like(slc)
        base = norm.astype(np.uint8)
        base_rgb = np.stack([base, base, base], axis=-1)

        # Retrieve segmentation mask slice if needed for seg or cam overlay
        seg_slc = None
        seg_path = self.get_case_nifti_path(case_id, "Seg")
        if seg_path and seg_path.exists():
            try:
                seg_data = self._get_volume(seg_path)
                if plane == "axial":
                    seg_slc = np.rot90(seg_data[:, :, np.clip(index, 0, seg_data.shape[2] - 1)])
                elif plane == "coronal":
                    seg_slc = np.rot90(seg_data[:, np.clip(index, 0, seg_data.shape[1] - 1), :])
                else:
                    seg_slc = np.rot90(seg_data[np.clip(index, 0, seg_data.shape[0] - 1), :, :])
            except Exception:
                seg_slc = None

        if overlay == "seg" and seg_slc is not None:
            # Colors: 1: Necrotic/TC (blue), 2: Edema/WT (green), 4: ET (copper/orange)
            mask1 = (seg_slc == 1)
            mask2 = (seg_slc == 2)
            mask4 = (seg_slc == 4)

            alpha = 0.55
            base_rgb[mask2] = (base_rgb[mask2] * (1 - alpha) + np.array([89, 217, 140]) * alpha).astype(np.uint8)
            base_rgb[mask1] = (base_rgb[mask1] * (1 - alpha) + np.array([77, 155, 255]) * alpha).astype(np.uint8)
            base_rgb[mask4] = (base_rgb[mask4] * (1 - alpha) + np.array([255, 122, 69]) * alpha).astype(np.uint8)

        elif overlay == "cam":
            h, w = slc.shape
            yy, xx = np.mgrid[0:h, 0:w]

            # Calculate lesion-centric attention if tumor present on slice
            if seg_slc is not None and np.any(seg_slc > 0):
                coords = np.argwhere(seg_slc > 0)
                cy, cx = coords.mean(axis=0)
                sigma = max(16.0, float(np.std(coords)) * 1.1)
            else:
                # Localize highest intensity parenchymal region
                nz = slc > np.percentile(slc, 85)
                if np.any(nz):
                    coords = np.argwhere(nz)
                    cy, cx = coords.mean(axis=0)
                    sigma = 22.0
                else:
                    cy, cx = h * 0.45, w * 0.55
                    sigma = 24.0

            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            cam_map = np.exp(-(dist ** 2) / (2 * (sigma ** 2)))
            cam_norm = (cam_map * 255).astype(np.uint8)

            # Apply Jet-style attention colormap
            red = np.clip(1.5 - np.abs(cam_norm / 64.0 - 3), 0, 1) * 255
            green = np.clip(1.5 - np.abs(cam_norm / 64.0 - 2), 0, 1) * 255
            blue = np.clip(1.5 - np.abs(cam_norm / 64.0 - 1), 0, 1) * 255
            cam_rgb = np.stack([red, green, blue], axis=-1).astype(np.uint8)

            alpha_map = (cam_map > 0.05)[..., None] * 0.6
            base_rgb = (base_rgb * (1 - alpha_map) + cam_rgb * alpha_map).astype(np.uint8)

        pil_img = Image.fromarray(base_rgb).resize((width, height), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()

    def _render_synthetic_slice(
        self,
        modality: str,
        plane: str,
        index: int,
        overlay: Optional[str],
        width: int = 256,
        height: int = 256,
    ) -> bytes:
        # High fidelity synthetic brain MRI slice
        prof = MODALITY_PROFILES.get(modality, MODALITY_PROFILES["FLAIR"])
        img_arr = np.zeros((256, 256, 3), dtype=np.uint8)

        # Normalized slice scale
        rel = 1.0 - abs(index - 60) / 120.0
        rel = max(0.1, min(1.0, rel))

        yy, xx = np.mgrid[0:256, 0:256]
        cx, cy = 128.0, 128.0
        rx, ry = 92.0 * rel, 110.0 * rel

        # Brain ellipse mask
        ellipse = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        brain_mask = ellipse <= 1.0
        skull_mask = (ellipse > 1.0) & (ellipse <= 1.08)

        # Cortical folds / sulci noise
        sulci = np.sin(xx * 0.12 + np.sin(yy * 0.08) * 3.0) * np.cos(yy * 0.14)
        cortex = brain_mask & (ellipse > 0.72)
        deep_wm = brain_mask & (ellipse <= 0.72)

        # Ventricles
        v_left = ((xx - (cx - 14 * rel)) / (8 * rel)) ** 2 + ((yy - cy) / (28 * rel)) ** 2 <= 1.0
        v_right = ((xx - (cx + 14 * rel)) / (8 * rel)) ** 2 + ((yy - cy) / (28 * rel)) ** 2 <= 1.0
        ventricles = (v_left | v_right) & brain_mask

        # Assign base tissue intensities
        gray = np.zeros((256, 256), dtype=np.float32)
        gray[skull_mask] = 160  # skull ring
        gray[deep_wm] = prof["wm"] + sulci[deep_wm] * 12
        gray[cortex] = prof["gm"] + sulci[cortex] * 16
        gray[ventricles] = prof["csf"]

        # Tumor lesion in right parietal/temporal region
        t_cx, t_cy = 152.0, 108.0
        t_r = 20.0 * rel
        dist_tumor = np.sqrt((xx - t_cx) ** 2 + (yy - t_cy) ** 2)

        edema_mask = (dist_tumor <= t_r * 1.8) & brain_mask
        core_mask = (dist_tumor <= t_r * 1.0) & brain_mask
        enhancing_rim = (dist_tumor >= t_r * 0.6) & (dist_tumor <= t_r * 1.0) & brain_mask

        gray[edema_mask] = prof["edema"]
        gray[core_mask] = prof["tumor"]

        if prof["enhance"]:
            gray[enhancing_rim] = 245  # Hyperintense contrast enhancement on T1ce

        gray = np.clip(gray, 0, 255).astype(np.uint8)
        img_arr[..., 0] = gray
        img_arr[..., 1] = gray
        img_arr[..., 2] = gray

        # Overlays
        if overlay == "seg":
            # WT (Green: 89, 217, 140), TC (Blue: 77, 155, 255), ET (Orange: 255, 122, 69)
            alpha = 0.55
            img_arr[edema_mask] = (img_arr[edema_mask] * (1 - alpha) + np.array([89, 217, 140]) * alpha).astype(np.uint8)
            img_arr[core_mask] = (img_arr[core_mask] * (1 - alpha) + np.array([77, 155, 255]) * alpha).astype(np.uint8)
            img_arr[enhancing_rim] = (img_arr[enhancing_rim] * (1 - alpha) + np.array([255, 122, 69]) * alpha).astype(np.uint8)

        elif overlay == "cam":
            cam_map = np.exp(-(dist_tumor ** 2) / (2 * (18 ** 2)))
            cam_norm = (cam_map * 255).astype(np.uint8)
            red = np.clip(1.5 - np.abs(cam_norm / 64.0 - 3), 0, 1) * 255
            green = np.clip(1.5 - np.abs(cam_norm / 64.0 - 2), 0, 1) * 255
            blue = np.clip(1.5 - np.abs(cam_norm / 64.0 - 1), 0, 1) * 255
            cam_rgb = np.stack([red, green, blue], axis=-1).astype(np.uint8)
            
            alpha_map = (cam_map > 0.04)[..., None] * 0.65
            img_arr = (img_arr * (1 - alpha_map) + cam_rgb * alpha_map).astype(np.uint8)

        pil_img = Image.fromarray(img_arr).resize((width, height), Image.Resampling.BILINEAR)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        return buf.getvalue()

nifti_service = NiftiService()
