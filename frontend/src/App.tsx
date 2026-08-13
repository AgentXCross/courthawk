import { useRef, useState } from 'react'
import { useVideoTime } from './hooks/useVideoTime'
import type { ChangeEvent, DragEvent } from 'react'
import './App.css'
import { analyzeVideo } from './api/analyze'
import type { PointAnalysis } from './types/analysis'
import MiniCourt from "./components/MiniCourt"
import VideoPlayer from './components/VideoPlayer'
import StatsTable from './components/StatsTable'

function App() {
  // UI state 
  const [analysis, setAnalysis] = useState<PointAnalysis | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const currentTime = useVideoTime(videoRef, analysis !== null)
  const currentFrame = analysis ? Math.floor(currentTime * analysis.fps) : 0

  // Reference to an HTML <input> element
  const fileInputRef = useRef<HTMLInputElement>(null)

  async function handleFile(file: File) { 
    // async function so when we await we continue with other operations while waiting for the analysis
    setIsLoading(true)
    setError(null)

    try {
      const result = await analyzeVideo(file)
      setAnalysis(result)
    } catch (err) {
      setError(err instanceof Error ? err.message: 'Analysis failed')
    } finally {
      setIsLoading(false)
    }
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    // Call this function when a file is dropped onto the video <div>
    event.preventDefault()
    const file = event.dataTransfer.files[0]
    if (file) handleFile(file)
  }

  function handleInputChange(event: ChangeEvent<HTMLInputElement>) {
    // Call this function when the user clicks to upload
    const file = event.target.files?.[0]
    if (file) handleFile(file)
  }

  return (
    <div id="layout">
      <nav id="navbar">COURTHAWK</nav>
      <div id="content">
        <section id="mini-court">
          <MiniCourt analysis={analysis} currentFrame={currentFrame}/>
        </section>
        <section id="video">
          {analysis ? (
            <VideoPlayer src={analysis.annotated_video_path} videoRef={videoRef}/>
          ) : (
            <div
              id="upload-dropzone"
              onDragOver={(event) => event.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="video/*"
                onChange={handleInputChange}
                hidden
              />
              <span id="upload-icon">{isLoading ? '⏳' : '⬆'}</span>
              <h2>{isLoading ? 'ANALYZING' : 'DROP YOUR POINT'}</h2>
              <p>{isLoading ? 'Breaking down every shot…' : 'Click or drag a video file to get started'}</p>
              {error && <p id="upload-error">{error}</p>}
            </div>
          )}
        </section>
      </div>
      <section id="stats">{analysis && <StatsTable />}</section>
    </div>
  )
}

export default App
