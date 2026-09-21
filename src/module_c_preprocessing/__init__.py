"""
NeuroVision AI: Module C - MRI Preprocessing & Patient-Level Splitting Package
Exports normalizers, spatial croppers, patient splitters, and end-to-end pipelines.
"""
from src.module_c_preprocessing.normalizer import IntensityNormalizer
from src.module_c_preprocessing.cropper import SpatialCropper
from src.module_c_preprocessing.splitter import PatientDataSplitter
from src.module_c_preprocessing.preprocessor import MRIPreprocessor
from src.module_c_preprocessing.pipeline import PreprocessingPipeline

__all__ = [
    "IntensityNormalizer",
    "SpatialCropper",
    "PatientDataSplitter",
    "MRIPreprocessor",
    "PreprocessingPipeline"
]
