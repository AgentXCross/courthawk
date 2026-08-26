import type { Shot, ShotType } from '../types/analysis'

interface ShotTrackProps {
  shots: Shot[]
  totalFrames: number
}

const SHOT_COLORS: Record<ShotType, string> = {
  forehand: '#ff9a0d',
  backhand: '#13c0ff',
  serve: '#2fff00',
  unknown: '#8a8a8a',
}

function ShotTrack({ shots, totalFrames }: ShotTrackProps) {
  return (
    <>
      <span className="timeline-row-label">Shots</span>
      <div className="timeline-row-plot shot-track">
        <div className="shot-track-line" />
        {shots.map((shot, i) => (
          <div
            key={i}
            className="shot-dot"
            style={{
              left: `${totalFrames > 0 ? (shot.frame / totalFrames) * 100 : 0}%`,
              background: SHOT_COLORS[shot.shot_type],
            }}
          />
        ))}
      </div>
    </>
  )
}

export default ShotTrack
