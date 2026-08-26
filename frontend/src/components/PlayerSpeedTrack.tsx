import type { CourtSide, PlayerSpeedSample } from '../types/analysis'
import { PLAYER_COLORS } from '../constants/playerColors'

interface PlayerSpeedTrackProps {
  playerIds: CourtSide[]
  playerSpeeds: PlayerSpeedSample[]
  totalFrames: number
}

function buildLinePoints(samples: { x: number; y: number }[]): string {
  return samples.map(({ x, y }) => `${x},${y}`).join(' ')
}

function PlayerSpeedTrack({ playerIds, playerSpeeds, totalFrames }: PlayerSpeedTrackProps) {
  const allSpeeds = playerSpeeds.flatMap((sample) => Object.values(sample.speeds_kmh))
  // +1 over the fastest identified speed, so the tallest step doesn't touch the top edge
  const axisMax = Math.max(1, ...allSpeeds) + 1

  return (
    <>
      <span className="timeline-row-label">Player Speeds</span>
      <div className="timeline-row-plot-wrapper">
        <span className="timeline-axis-label max">{Math.round(axisMax)}</span>
        <span className="timeline-axis-label min">0</span>
        <svg viewBox="0 0 400 60" preserveAspectRatio="none">
          <line x1={0} y1={60} x2={400} y2={60} stroke="#4a4a4c" strokeWidth={1} />
          {playerIds.map((playerId) => {
            const samples = playerSpeeds
              .map((sample) => {
                const speed = sample.speeds_kmh[playerId]
                if (speed === undefined) return null
                const x = totalFrames > 0 ? (sample.frame / totalFrames) * 400 : 0
                const y = 60 - (speed / axisMax) * 60
                return { x, y }
              })
              .filter((point): point is { x: number; y: number } => point !== null)

            return (
              <polyline
                key={playerId}
                points={buildLinePoints(samples)}
                fill="none"
                stroke={PLAYER_COLORS[playerId]}
                strokeWidth={0.7}
              />
            )
          })}
        </svg>
      </div>
    </>
  )
}

export default PlayerSpeedTrack
