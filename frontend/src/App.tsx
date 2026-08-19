import { useRef, useState } from 'react'
import { useVideoTime } from './hooks/useVideoTime'
import type { ChangeEvent, DragEvent, MouseEvent } from 'react'
import './App.css'
import { analyzeVideo, getSampleAnalysis } from './api/analyze'
import type { PointAnalysis } from './types/analysis'
import MiniCourt from "./components/MiniCourt"
import VideoPlayer from './components/VideoPlayer'
import AnalysisTimeline from './components/AnalysisTimeline'
import courthawkLogo from './assets/courthawk_logo.png'

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

  async function handleTrySample(event: MouseEvent) {
    // Stop this click from also triggering the dropzone's own onClick (which opens the file picker)
    event.stopPropagation()

    setIsLoading(true)
    setError(null)

    try {
      const result = await getSampleAnalysis()
      setAnalysis(result)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load sample')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div id="layout">
      <nav id="navbar">
        <img id="navbar-logo" src={courthawkLogo} alt="CourtHawk logo" />
        <a id="navbar-github" href="https://github.com/AgentXCross/courthawk" target="_blank" rel="noopener noreferrer">
          GitHub
        </a>
        <div id="navbar-title">CourtHawk</div>
        <div id="navbar-subtitle">Computer Vision Tennis System</div>
      </nav>
      <section id="hero">
        <p id="hero-status">Currently in development.</p>
      </section>
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
              <h2>{isLoading ? 'Running your point through CourtHawk' : 'Drop your point'}</h2>
              <p>{isLoading ? 'Breaking down every shot…' : 'Click or drag a video file to get started'}</p>
              {!isLoading && (
                <button id="try-sample-button" onClick={handleTrySample}>
                  Try this sample
                </button>
              )}
              {!isLoading && (
                <ul id="upload-requirements">
                  <li>Footage must be from the standard broadcast TV camera angle</li>
                  <li>Must contain one point only starting at the serve and ending when the point finishes</li>
                  <li>Both players visible and ready from the very first frame</li>
                  <li>Camera must be stationary during the duration of the point</li>
                </ul>
              )}
              {error && <p id="upload-error">{error}</p>}
            </div>
          )}
        </section>
      </div>
      <section id="stats">
        <AnalysisTimeline analysis={analysis} currentFrame={currentFrame} />
      </section>
    </div>
  )
}

export default App
