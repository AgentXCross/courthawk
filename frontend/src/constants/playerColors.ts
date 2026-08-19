import type { CourtSide } from '../types/analysis'

// Single source of truth for player-side colors, shared by MiniCourt and PlayerSpeedTrack
// so the two stay in sync.
export const PLAYER_COLORS: Record<CourtSide, string> = {
  close: 'rgb(0, 230, 90)',
  far: 'rgb(236, 0, 140)',
}
