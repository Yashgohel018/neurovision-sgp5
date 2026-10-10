import LandingNavbar from '@/components/landing/LandingNavbar'
import HeroSection from '@/components/landing/HeroSection'
import MultimodalSignalsSection from '@/components/landing/MultimodalSignalsSection'
import InteractiveMRIViewerSection from '@/components/landing/InteractiveMRIViewerSection'
import PipelineFlowSection from '@/components/landing/PipelineFlowSection'
import ThreeDVisualizationSection from '@/components/landing/ThreeDVisualizationSection'
import TumorVolumetrySection from '@/components/landing/TumorVolumetrySection'
import ModelInferenceSection from '@/components/landing/ModelInferenceSection'
import ExplainableAISection from '@/components/landing/ExplainableAISection'
import ClinicalAssistantSection from '@/components/landing/ClinicalAssistantSection'
import CaseManagementPreviewSection from '@/components/landing/CaseManagementPreviewSection'
import DetailedReportSection from '@/components/landing/DetailedReportSection'
import RoadAheadSection from '@/components/landing/RoadAheadSection'
import LandingFooter from '@/components/landing/LandingFooter'

export default function Home() {
  return (
    <div className="min-h-screen bg-[#03060d] text-white selection:bg-nv-cool selection:text-white">
      <LandingNavbar />
      <main>
        <HeroSection />
        <MultimodalSignalsSection />
        <InteractiveMRIViewerSection />
        <PipelineFlowSection />
        <ThreeDVisualizationSection />
        <TumorVolumetrySection />
        <ModelInferenceSection />
        <ExplainableAISection />
        <ClinicalAssistantSection />
        <CaseManagementPreviewSection />
        <DetailedReportSection />
        <RoadAheadSection />
      </main>
      <LandingFooter />
    </div>
  )
}
