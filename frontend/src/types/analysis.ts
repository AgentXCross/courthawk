// Matches the classes used in the POST /analyze response from engine.py. POST is an HTTP request method.
// The dict keys (player_id) are JSON string keys, even though the Python dicts used int.

export interface Point { // interface describes the shape an object must have, export allows other modules to use it
    x: number
    y: number
}

export type ShotType = 'serve' | 'forehand' | 'backhand' | 'unknown'

export interface Shot {
    frame: number
    player_id: number
    shot_type: ShotType
    ball_speed_kmh: number
}

export interface PlayerSpeedSample {
    frame: number
    speeds_kmh: Record<string, number> // player_id (as string) -> speed, Record is essentially a dictionary
}

export interface PointAnalysis {
    annotated_video_path: string // URL, not a filesystem path
    fps: number

    real_court_keypoints: Point[] // fixed 14-point court layout in meters
    player_ids: number[]
    player_court_foot_positions: Record<string, Point>[] // per frame, player_id (as string) -> position
    ball_court_positions: Point[] // per frame

    shots: Shot[]
    player_speeds: PlayerSpeedSample[]
}
