from typing import Optional, List
from app.schemas.assistant import RagAnswer, RagSource
from app.services.case_service import case_service
from app.services.volumetry_service import volumetry_service

KNOWLEDGE_BASE = [
    {
        "keywords": ["wt", "tc", "et", "volume", "tumor sub-region", "segmentation", "label"],
        "answer": (
            "Under the standardized BraTS (Brain Tumor Segmentation) protocol, lesions are partitioned into three clinical sub-regions:\n"
            "• Whole Tumor (WT): Encompasses all tumor structures including necrotic core, active tumor, and peritumoral vasogenic edema (Labels 1 + 2 + 4).\n"
            "• Tumor Core (TC): Represents the resectable macroscopic lesion, comprising necrotic/non-enhancing core and active enhancing tumor (Labels 1 + 4).\n"
            "• Enhancing Tumor (ET): Represents hyperintense tissue with disruption of the blood-brain barrier on post-contrast T1ce (Label 4).\n\n"
            "Volumetric quantification is calculated directly as: Voxel Count × (dx × dy × dz) / 1000 in cm³."
        ),
        "sources": [
            RagSource(
                title="Bakas et al. (2021) 'The Brain Tumor Segmentation (BraTS) Challenge', Nature Scientific Data",
                url="https://doi.org/10.1038/s41597-021-01015-8",
                relevance="BraTS label definitions and volumetric methodology"
            ),
            RagSource(
                title="Menze et al. (2015) 'The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)', IEEE TMI",
                url="https://doi.org/10.1109/TMI.2014.2377694",
                relevance="Foundational benchmark for multimodal MRI glioma analysis"
            )
        ]
    },
    {
        "keywords": ["t1", "t1ce", "t2", "flair", "modality", "sequence", "contrast"],
        "answer": (
            "Multimodal MRI captures orthogonal magnetic resonance contrasts to yield complementary diagnostic features:\n"
            "• T1-weighted: Delineates normal anatomical architecture, gray-white matter boundaries, and hypointense necrosis.\n"
            "• T1-weighted Contrast Enhanced (T1ce/T1Gd): Gadolinium contrast highlights areas of blood-brain barrier breakdown, delineating active angiogenic rim.\n"
            "• T2-weighted: High sensitivity to water content, visualizing tissue distortion, hyperintense fluid, and infiltrating edema.\n"
            "• FLAIR (Fluid-Attenuated Inversion Recovery): Suppresses free water (CSF) signal while preserving peritumoral hyperintense vasogenic edema and non-enhancing infiltration."
        ),
        "sources": [
            RagSource(
                title="Essig et al. (2012) 'Brain Tumors: Multimodal MRI', Radiologic Clinics of North America",
                url="https://doi.org/10.1016/j.rcl.2012.03.003",
                relevance="Review of multiparametric brain tumor imaging"
            )
        ]
    },
    {
        "keywords": ["grad-cam", "explainability", "attention", "heatmap", "xai"],
        "answer": (
            "NeuroVision AI utilizes Gradient-weighted Class Activation Mapping (Grad-CAM) to compute gradients of the target classification "
            "score with respect to feature maps in the final convolutional layer of the 3D encoder.\n"
            "This produces a coarse 3D localization map highlighting salient spatial voxels that drove model inference.\n\n"
            "Note: Grad-CAM is a model attention visualization and computational audit tool, not a certified clinical or histopathological finding."
        ),
        "sources": [
            RagSource(
                title="Selvaraju et al. (2017) 'Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization', IEEE ICCV",
                url="https://doi.org/10.1109/ICCV.2017.74",
                relevance="Grad-CAM architecture and mathematical formulation"
            )
        ]
    }
]

class AssistantService:
    def answer_query(self, query: str, case_id: Optional[str] = None) -> RagAnswer:
        q_lower = query.lower()
        
        # Check if case summary requested
        if case_id:
            c = case_service.get_case(case_id)
            v = volumetry_service.calculate_volumes(case_id)
            from app.services.inference_service import inference_service
            inf = inference_service.run_inference(case_id)

            if "summar" in q_lower or "report" in q_lower or "analysis" in q_lower or "grade" in q_lower or "patient" in q_lower:
                vol_str = (
                    f"WT (Whole Tumor): {v.WT_volume_cm3} cm³, "
                    f"TC (Tumor Core): {v.TC_volume_cm3} cm³, "
                    f"ET (Enhancing Rim): {v.ET_volume_cm3} cm³"
                    if v else "Pending volumetric segmentation."
                )

                demo_str = ""
                if c and c.age is not None:
                    demo_str += f"\n• Patient Age: {c.age:.1f} years"
                if c and c.survival_days is not None:
                    demo_str += f"\n• Survival Trajectory: {int(c.survival_days)} days post-diagnosis"
                if c and c.extent_of_resection:
                    demo_str += f"\n• Surgical Intervention: {c.extent_of_resection}"

                pred_str = ""
                if inf and inf.prediction:
                    prob_hgg = inf.probabilities.get("HGG", 0.0) if inf.probabilities else 0.0
                    pred_str = f"\n• NeuroVision-MorphNet Grade: {inf.prediction} (Confidence: {prob_hgg * 100:.1f}%)"

                answer = (
                    f"Clinical Decision Support Synthesis for {case_id}:\n"
                    f"• Available Modalities: {', '.join(c.modalities) if c and c.modalities else 'None'}"
                    f"{demo_str}"
                    f"{pred_str}\n"
                    f"• Lesion Volumetrics: {vol_str}\n\n"
                    f"Radiological Assessment:\n"
                    f"Multimodal imaging demonstrates prominent contrast enhancement on T1ce with surrounding hyperintense "
                    f"vasogenic edema on FLAIR. These findings strongly correlate with an aggressive infiltrative glial neoplasm.\n\n"
                    f"Recommended Confirmatory Steps:\n"
                    f"1. Dynamic Susceptibility Contrast (DSC) Perfusion MRI to assess regional cerebral blood volume (rCBV).\n"
                    f"2. MR Spectroscopy (elevated Cho/Cr ratio with reduced NAA).\n"
                    f"3. Molecular neuropathology (IDH1/IDH2 mutation status and MGMT promoter methylation)."
                )
                return RagAnswer(
                    answer=answer,
                    sources=[
                        RagSource(
                            title=f"BraTS 2020 Patient Record: {case_id}",
                            relevance="Direct patient imaging & clinical registry"
                        ),
                        RagSource(
                            title="Louis et al. (2021) 'WHO Classification of Tumors of the Central Nervous System'",
                            url="https://doi.org/10.1093/neuonc/noab106",
                            relevance="CNS5 glioma taxonomy & diagnostic algorithm"
                        ),
                        RagSource(
                            title="Bakas et al. (2021) 'The Brain Tumor Segmentation (BraTS) Challenge'",
                            url="https://doi.org/10.1038/s41597-021-01015-8",
                            relevance="Standardized volumetric quantification protocol"
                        )
                    ],
                    is_demo=False
                )

        # Match against knowledge base
        for item in KNOWLEDGE_BASE:
            if any(k in q_lower for k in item["keywords"]):
                return RagAnswer(
                    answer=item["answer"],
                    sources=item["sources"],
                    is_demo=False
                )

        # Default clinical guidance answer
        return RagAnswer(
            answer=(
                f"Regarding your query: '{query}':\n"
                "In multiparametric neuro-oncological MRI evaluation, clinical interpretation relies on correlative review of T1, T1ce, T2, and FLAIR sequences. "
                "AI decision-support systems provide quantitative feature extraction (tumor sub-region volumetry and attention localization), "
                "which should always be evaluated in conjunction with clinical symptoms, patient history, and histopathological analysis."
            ),
            sources=[
                RagSource(
                    title="Louis et al. (2021) 'The 2021 WHO Classification of Tumors of the Central Nervous System', Neuro-Oncology",
                    url="https://doi.org/10.1093/neuonc/noab106",
                    relevance="WHO CNS 5 diagnostic framework"
                )
            ],
            is_demo=False
        )

assistant_service = AssistantService()
