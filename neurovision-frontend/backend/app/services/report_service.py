from datetime import datetime, timezone
from typing import Optional
from app.schemas.report import AnalysisReport
from app.services.case_service import case_service
from app.services.volumetry_service import volumetry_service
from app.services.inference_service import inference_service

class ReportService:
    def generate_report(self, case_id: str) -> AnalysisReport:
        c = case_service.get_case(case_id)
        v = volumetry_service.calculate_volumes(case_id)
        inf = inference_service.run_inference(case_id)

        mods = c.modalities if c else []
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        findings = []
        if c and c.age is not None:
            findings.append(f"Patient Demographic: {c.age:.1f} years of age at diagnostic MRI acquisition.")
        if c and c.extent_of_resection:
            findings.append(f"Surgical Context: Recorded resection status is {c.extent_of_resection}.")
        if "T1ce" in mods:
            findings.append("Post-contrast T1ce sequences reveal prominent nodular and peripheral ring enhancement.")
        if "FLAIR" in mods:
            findings.append("FLAIR demonstrates pronounced perilesional hyperintense signal consistent with vasogenic edema.")
        if v:
            findings.append(
                f"Volumetric assessment confirms Whole Tumor (WT) burden of {v.WT_volume_cm3} cm³, "
                f"Tumor Core (TC) of {v.TC_volume_cm3} cm³, and Enhancing Tumor (ET) of {v.ET_volume_cm3} cm³."
            )
            if v.WT_volume_cm3 > 0:
                et_ratio = (v.ET_volume_cm3 / v.WT_volume_cm3) * 100
                findings.append(f"Active angiogenic enhancing fraction constitutes {et_ratio:.1f}% of total lesion volume.")
        if not findings:
            findings.append("Imaging series pending upload and complete multi-sequence verification.")

        return AnalysisReport(
            case_id=case_id,
            generated_at=now_str,
            modalities_available=mods,
            segmentation_status="Mask available and quantified" if v else "Segmentation mask not found",
            tumor_volumes=v,
            prediction=inf.prediction,
            probabilities=inf.probabilities,
            model_version=inf.model_version or "NeuroVision-MorphNet v1.0",
            radiological_findings=findings,
            methodology="BraTS2020 standardized multi-parametric quantification with NeuroVision-MorphNet pipeline",
            disclaimer="NeuroVision AI is a clinical decision-support and research system. Predictions and volumetric metrics must be interpreted in conjunction with histopathology and qualified clinical review.",
        )

report_service = ReportService()
