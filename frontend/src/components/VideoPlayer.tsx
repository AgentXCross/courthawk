import { useEffect, useState } from 'react'
import type { ChangeEvent, RefObject } from 'react'
import { API_BASE_URL } from '../api/config'
import { useVideoTime } from '../hooks/useVideoTime'

interface VideoPlayerProps {
  src: string
  videoRef: RefObject<HTMLVideoElement | null>
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds)) return '0:00'
  const total = Math.floor(seconds)
  const mins = Math.floor(total / 60)
  const secs = total % 60
  return `${mins}:${secs.toString().padStart(2, '0')}`
}

// Native <video controls> can't be restyled reliably (Chrome's ::-webkit-media-controls-*
// hooks are non-standard and being phased out, Firefox exposes none at all), so this builds
// a custom play button + seek bar + time display instead, using the same currentTime
// tracking (useVideoTime) the mini-court/timeline already sync to.
function VideoPlayer({ src, videoRef }: VideoPlayerProps) {
  const currentTime = useVideoTime(videoRef, true)
  const [duration, setDuration] = useState(0)
  const [isPlaying, setIsPlaying] = useState(false)

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const handleLoadedMetadata = () => setDuration(video.duration)
    const handlePlay = () => setIsPlaying(true)
    const handlePause = () => setIsPlaying(false)

    video.addEventListener('loadedmetadata', handleLoadedMetadata)
    video.addEventListener('play', handlePlay)
    video.addEventListener('pause', handlePause)

    if (video.readyState >= 1) setDuration(video.duration)

    return () => {
      video.removeEventListener('loadedmetadata', handleLoadedMetadata)
      video.removeEventListener('play', handlePlay)
      video.removeEventListener('pause', handlePause)
    }
  }, [videoRef, src])

  function togglePlay() {
    const video = videoRef.current
    if (!video) return
    if (video.paused) video.play()
    else video.pause()
  }

  function handleSeek(event: ChangeEvent<HTMLInputElement>) {
    const video = videoRef.current
    if (!video) return
    video.currentTime = Number(event.target.value)
  }

  const progressPct = duration > 0 ? (currentTime / duration) * 100 : 0

  return (
    <div id="video-player">
      <video ref={videoRef} src={`${API_BASE_URL}${src}`} />
      <div id="video-controls">
        <button id="video-play-button" onClick={togglePlay} type="button">
          {isPlaying ? '⏸' : '▶'}
        </button>
        <input
          id="video-seek-bar"
          type="range"
          min={0}
          max={duration || 0}
          step={0.01}
          value={currentTime}
          onChange={handleSeek}
          style={{ background: `linear-gradient(to right, #c7ff00 ${progressPct}%, #3a3a3c ${progressPct}%)` }}
        />
        <span id="video-time-display">
          {formatTime(currentTime)} / {formatTime(duration)}
        </span>
      </div>
    </div>
  )
}

export default VideoPlayer
