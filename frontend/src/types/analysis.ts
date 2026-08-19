// Matches the classes used in the POST /analyze response from engine.py.

export interface Point { // interface describes the shape an object must have, export allows other modules to use it
    x: number
    y: number
}

export type ShotType = 'serve' | 'forehand' | 'backhand' | 'unknown'

// Players are identified by which side of the court they're on
export type CourtSide = 'close' | 'far'

export interface Shot {
    frame: number
    player_id: CourtSide
    shot_type: ShotType
    ball_speed_kmh: number
}

export interface PlayerSpeedSample {
    frame: number
    speeds_kmh: Record<CourtSide, number>
}

export interface PointAnalysis {
    annotated_video_path: string // URL, not a filesystem path
    fps: number

    real_court_keypoints: Point[] // fixed 14-point court layout in meters
    player_ids: CourtSide[]
    player_court_foot_positions: Record<CourtSide, Point>[] // per frame
    ball_court_positions: Point[] // per frame

    shots: Shot[]
    player_speeds: PlayerSpeedSample[]
}
