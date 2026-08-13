import type { RefObject } from 'react'
import { API_BASE_URL } from '../api/config'

interface VideoPlayerProps {
  src: string
  videoRef: RefObject<HTMLVideoElement | null>
}

function VideoPlayer({ src, videoRef }: VideoPlayerProps) {
  return <video ref={videoRef} src={`${API_BASE_URL}${src}`} controls />
}

export default VideoPlayer
