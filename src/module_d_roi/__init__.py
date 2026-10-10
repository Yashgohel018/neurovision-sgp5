"""
NeuroVision AI: Module D - 3D Tumor Segmentation & ROI Extraction Package
"""
from src.module_d_roi.bbox_extractor import TumorBoundingBox, TumorBoundingBoxExtractor
from src.module_d_roi.roi_cropper import TumorROICropper
from src.module_d_roi.segmentor import (
    BaseTumorSegmentor,
    GroundTruthSegmentor,
    HeuristicFLAIRSegmentor,
    DeepTumorSegmentor,
    get_segmentor
)
from src.module_d_roi.roi_metadata import ROIMetadataGenerator
from src.module_d_roi.visualizer import ROIVisualizer
from src.module_d_roi.pipeline import ROIExtractionPipeline

from src.module_d_roi.attention_gate import (
    AttentionGate3D,
    AttentionGatedUNETR,
    AttentionGatedROISegmentor,
)

__all__ = [
    "TumorBoundingBox",
    "TumorBoundingBoxExtractor",
    "TumorROICropper",
    "BaseTumorSegmentor",
    "GroundTruthSegmentor",
    "HeuristicFLAIRSegmentor",
    "DeepTumorSegmentor",
    "AttentionGate3D",
    "AttentionGatedUNETR",
    "AttentionGatedROISegmentor",
    "get_segmentor",
    "ROIMetadataGenerator",
    "ROIVisualizer",
    "ROIExtractionPipeline",
]
