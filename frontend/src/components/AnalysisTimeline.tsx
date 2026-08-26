import type { PointAnalysis } from '../types/analysis'
import ShotTrack from './ShotTrack'
import PlayerSpeedTrack from './PlayerSpeedTrack'
import PanelCorners from './PanelCorners'

interface AnalysisTimelineProps {
  analysis: PointAnalysis | null
  currentFrame: number
}

function AnalysisTimeline({ analysis, currentFrame }: AnalysisTimelineProps) {
  // Dense array, one entry per video frame, so its length is the point's total frame count
  const totalFrames = analysis ? analysis.player_court_foot_positions.length : 0
  const progress = totalFrames > 0 ? Math.min(1, currentFrame / totalFrames) : 0

  return (
    <div id="analysis-timeline" className="panel">
      <PanelCorners />
      <ShotTrack shots={analysis?.shots ?? []} totalFrames={totalFrames} />
      <PlayerSpeedTrack
        playerIds={analysis?.player_ids ?? []}
        playerSpeeds={analysis?.player_speeds ?? []}
        totalFrames={totalFrames}
      />
      {analysis && (
        <div
          id="timeline-cursor-line"
          style={{ left: `calc(90px + 1rem + (100% - 90px - 1rem) * ${progress})` }}
        />
      )}
    </div>
  )
}

export default AnalysisTimeline
