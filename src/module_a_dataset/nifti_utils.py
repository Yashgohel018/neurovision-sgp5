"""
NIfTI inspection and spatial metadata extraction utilities.
Optimized to parse NIfTI headers without loading voxel data arrays into memory.
"""
import os
from typing import Dict, Any, Tuple
# pyrefly: ignore [missing-import]
import nibabel as nib
import numpy as np


def safe_load_header(file_path: str) -> Tuple[bool, Any, str]:
    """
    Safely load a NIfTI file header without loading the entire 3D array into RAM.
    
    Args:
        file_path: Path to the .nii or .nii.gz file.
        
    Returns:
        Tuple of (is_valid, nifti_img_proxy, error_message).
    """
    if not os.path.isfile(file_path):
        return False, None, f"File not found: {file_path}"
    
    try:
        # nib.load loads file proxy & header lazily; array voxels are NOT loaded into RAM.
        img = nib.load(file_path)
        # Touch header to verify integrity
        _ = img.header
        return True, img, ""
    except Exception as e:
        return False, None, f"Corrupted or invalid NIfTI file: {str(e)}"


def get_nifti_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract comprehensive spatial and technical metadata from a NIfTI file header.
    
    Metadata extracted:
    - shape: (dim_x, dim_y, dim_z)
    - voxel_spacing: (sx, sy, sz) in millimeters
    - data_type: numerical type stored in the header (e.g., int16)
    - orientation: 3-letter anatomical coordinate orientation (e.g., 'LPS', 'RAS')
    - affine: 4x4 coordinate transformation matrix from voxel indices to scanner space
    - file_size_mb: physical file size in Megabytes
    - is_valid: boolean indicating header readability
    - error: any error message encountered
    
    Args:
        file_path: Path to NIfTI file (.nii or .nii.gz)
        
    Returns:
        Dictionary containing metadata fields.
    """
    is_valid, img, err = safe_load_header(file_path)
    if not is_valid:
        return {
            "file_path": file_path,
            "exists": os.path.exists(file_path),
            "is_valid": False,
            "shape": None,
            "voxel_spacing": None,
            "data_type": None,
            "orientation": None,
            "affine": None,
            "file_size_mb": os.path.getsize(file_path) / (1024 * 1024) if os.path.exists(file_path) else 0.0,
            "error": err,
        }
    
    header = img.header
    shape = tuple(int(dim) for dim in img.shape[:3])
    
    # Header zooms give spatial resolution in mm
    zooms = tuple(float(z) for z in header.get_zooms()[:3])
    
    # Affine matrix (4x4)
    affine = img.affine
    
    # Anatomical orientation: e.g., ('L', 'P', 'S') -> 'LPS'
    try:
        axcodes = "".join(nib.aff2axcodes(affine))
    except Exception:
        axcodes = "UNKNOWN"
        
    data_type = str(header.get_data_dtype())
    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    
    return {
        "file_path": file_path,
        "exists": True,
        "is_valid": True,
        "shape": shape,
        "voxel_spacing": zooms,
        "data_type": data_type,
        "orientation": axcodes,
        "affine": affine.tolist(),
        "file_size_mb": round(file_size_mb, 2),
        "error": None,
    }
