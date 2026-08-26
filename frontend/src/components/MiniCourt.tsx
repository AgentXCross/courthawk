import type { CourtSide, Point, PointAnalysis } from '../types/analysis'
import { PLAYER_COLORS } from '../constants/playerColors'

interface MiniCourtProps {
  analysis: PointAnalysis | null
  currentFrame: number
}

const PADDING = 4 // meters of extra space around the court, for players standing behind the baseline

// Mirrors engine.py's _real_court_keypoints(). That function only ever depends on fixed
// court-dimension constants, never on the video, so this layout is identical on every analysis.
// Draw it before any upload happens, so the court is always visible.
const DOUBLE_LINE_WIDTH = 10.973
const BASELINE_TO_NET = 11.89
const DOUBLE_ALLEY_WIDTH = 1.372
const BASELINE_TO_SERVICE = 5.49

// Pairs of indices into the 14-point keypoints array (see the order in engine.py's
// _real_court_keypoints()). Each pair is one line segment of the court.
const COURT_LINES: [number, number][] = [
  [0, 1], // top baseline
  [2, 3], // bottom baseline
  [0, 2], // left doubles sideline
  [1, 3], // right doubles sideline
  [4, 5], // left singles sideline
  [6, 7], // right singles sideline
  [8, 9], // top service line
  [10, 11], // bottom service line
  [12, 13], // center service line
]

function defaultCourtKeypoints(): Point[] {
  // Returns a list of points 
  const width = DOUBLE_LINE_WIDTH
  const length = BASELINE_TO_NET * 2
  const alley = DOUBLE_ALLEY_WIDTH
  const service = BASELINE_TO_SERVICE

  return [
    { x: 0, y: 0 },
    { x: width, y: 0 },
    { x: 0, y: length },
    { x: width, y: length },
    { x: alley, y: 0 },
    { x: alley, y: length },
    { x: width - alley, y: 0 },
    { x: width - alley, y: length },
    { x: alley, y: service },
    { x: width - alley, y: service },
    { x: alley, y: length - service },
    { x: width - alley, y: length - service },
    { x: width / 2, y: service },
    { x: width / 2, y: length - service },
  ]
}

function MiniCourt({ analysis, currentFrame }: MiniCourtProps) {
  const keypoints = analysis?.real_court_keypoints ?? defaultCourtKeypoints()
  const width = Math.max(...keypoints.map((p) => p.x))
  const length = Math.max(...keypoints.map((p) => p.y))

  const playerPositions = analysis?.player_court_foot_positions[currentFrame] ?? {}
  const ballPosition = analysis?.ball_court_positions[currentFrame]

  return (
    <svg
      id="mini-court-inner"
      viewBox={`${-PADDING} ${-PADDING} ${width + 2 * PADDING} ${length + 2 * PADDING}`}
      preserveAspectRatio="xMidYMid slice"
    >
      <defs>
        <filter id="mini-court-line-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="0.3" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      <rect x={-PADDING} y={-PADDING} width={width + 2 * PADDING} height={length + 2 * PADDING} fill="#000000" /> {/*#63ab64 for green*/}
      <rect x={0} y={0} width={width} height={length} fill="#000000" />

      <g filter="url(#mini-court-line-glow)">
        {COURT_LINES.map(([a, b], i) => (
          <line
            key={i}
            x1={keypoints[a].x}
            y1={keypoints[a].y}
            x2={keypoints[b].x}
            y2={keypoints[b].y}
            stroke="white"
            strokeWidth={0.15}
          />
        ))}

        <line x1={0} y1={length / 2} x2={width} y2={length / 2} stroke="white" strokeWidth={0.15} />
      </g>

      {keypoints.map((point, i) => (
        <circle key={i} cx={point.x} cy={point.y} r={0.1} fill="white" />
      ))}

      {analysis &&
        (Object.entries(playerPositions) as [CourtSide, Point][]).map(([side, point]) => (
          <circle key={side} cx={point.x} cy={point.y} r={0.4} fill={PLAYER_COLORS[side]} />
        ))}

      {analysis && ballPosition && <circle cx={ballPosition.x} cy={ballPosition.y} r={0.3} fill="#CCFF00" />}
    </svg>
  )
}

export default MiniCourt
