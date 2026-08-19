import type { CourtSide, PlayerSpeedSample } from '../types/analysis'

interface PlayerSpeedTrackProps {
  playerIds: CourtSide[]
  playerSpeeds: PlayerSpeedSample[]
  totalFrames: number
}

const PLAYER_COLORS = ['#DF1AA0', '#00ADF2']

// Step ("staircase") line instead of connecting points directly — gives a blocky,
// bar-chart-like silhouette while still being a thin stroked line, not filled bars.
function buildStepPoints(samples: { x: number; y: number }[]): string {
  if (samples.length === 0) return ''

  const parts = [`${samples[0].x},${samples[0].y}`]
  for (let i = 1; i < samples.length; i++) {
    const prev = samples[i - 1]
    const curr = samples[i]
    parts.push(`${curr.x},${prev.y}`) // horizontal segment at the previous height
    parts.push(`${curr.x},${curr.y}`) // vertical segment up/down to the new height
  }
  return parts.join(' ')
}

function PlayerSpeedTrack({ playerIds, playerSpeeds, totalFrames }: PlayerSpeedTrackProps) {
  const allSpeeds = playerSpeeds.flatMap((sample) => Object.values(sample.speeds_kmh))
  // +5 over the fastest identified speed, so the tallest step doesn't touch the top edge
  const axisMax = Math.max(1, ...allSpeeds) + 5

  return (
    <>
      <span className="timeline-row-label">Player Speeds</span>
      <div className="timeline-row-plot-wrapper">
        <span className="timeline-axis-label max">{Math.round(axisMax)}</span>
        <span className="timeline-axis-label min">0</span>
        <svg viewBox="0 0 400 60" preserveAspectRatio="none">
          <line x1={0} y1={60} x2={400} y2={60} stroke="#4a4a4c" strokeWidth={1} />
          {playerIds.map((playerId, index) => {
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
                points={buildStepPoints(samples)}
                fill="none"
                stroke={PLAYER_COLORS[index] ?? 'gray'}
                strokeWidth={1.2}
              />
            )
          })}
        </svg>
      </div>
    </>
  )
}

export default PlayerSpeedTrack
