"""
Module A: Dataset Acquisition, Indexing, and Validation.
Handles BraTS 3D MRI scanning, modality path mapping, label linking, and spatial integrity checks.
"""
from .nifti_utils import get_nifti_metadata, safe_load_header
from .indexer import BraTSIndexer
from .validator import BraTSValidator

__all__ = [
    "get_nifti_metadata",
    "safe_load_header",
    "BraTSIndexer",
    "BraTSValidator",
]
