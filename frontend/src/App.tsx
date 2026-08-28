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
import PanelCorners from './components/PanelCorners'

function App() {
  // UI state 
  const [analysis, setAnalysis] = useState<PointAnalysis | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [videoReady, setVideoReady] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const videoRef = useRef<HTMLVideoElement>(null)
  const currentTime = useVideoTime(videoRef, analysis !== null)
  const currentFrame = analysis ? Math.floor(currentTime * analysis.fps) : 0

  // Reference to an HTML <input> element
  const fileInputRef = useRef<HTMLInputElement>(null)

  async function handleFile(file: File) {
    // async function so when we await we continue with other operations while waiting for the analysis
    setIsLoading(true)
    setVideoReady(false)
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
    setVideoReady(false)
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
        <div id="navbar-brand">
          <img id="navbar-logo" src={courthawkLogo} alt="CourtHawk logo" />
          <span id="navbar-title">CourtHawk</span>
        </div>
        <a id="navbar-github" href="https://github.com/AgentXCross/courthawk" target="_blank" rel="noopener noreferrer">
          <svg viewBox="0 0 16 16" width="16" height="16" fill="currentColor" aria-hidden="true">
            <path fillRule="evenodd" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
          </svg>
          GitHub
        </a>
      </nav>
      <p id="ram-warning">
        Warning: Backend is hosted on Render free-tier with only 512 MB of RAM. 
        The models themselves require at least 1 GB so uploading your own video will not work on 
        the web app, though will work if the project is run locally. The "Try Sample" feature won't hit memory limit since
        its outputs are pre-computed.
      </p>
      <div id="content">
        <aside id="sidebar" className="panel">
          <PanelCorners />
          <h3 className="panel-title">About</h3>
          <p id="sidebar-text">
            CourtHawk is a computer vision system for tennis that detects and tracks players and ball, 
            estimates player poses, classifies shots, detects court geometry, and projects movements onto a bird's-eye-view using homography.
          </p>

          <p id="sidebar-text">
            As input, CourtHawk expects the video to only contain one point starting at the serve and ending when the point finishes. 
            The footage must be from the standard broadcast TV camera angle. Both players visible and ready from the very first frame
            and the camera cannot move during the duration of the point.
          </p>

          <p id="sidebar-text">
            For a full analysis of the pipeline, read the README.md markdown file on GitHub. CourtHawk is 
            currently under development with many upcoming changes.
          </p>
        </aside>
        <section id="mini-court" className="panel">
          <PanelCorners />
          <h3 className="panel-title">Bird's-Eye View</h3>
          <div className="panel-content">
            <MiniCourt analysis={analysis} currentFrame={currentFrame}/>
          </div>
        </section>
        <section id="video" className="panel">
          <PanelCorners />
          <h3 className="panel-title">Point Video</h3>
          <div className="panel-content">
            {analysis ? (
              <div id="video-wrapper">
                <VideoPlayer src={analysis.annotated_video_path} videoRef={videoRef} onReady={() => setVideoReady(true)}/>
                {!videoReady && (
                  <div id="video-loading-overlay">
                    <span className="dot-spinner">
                      <span></span>
                      <span></span>
                      <span></span>
                    </span>
                    <h2>Loading Video...</h2>
                    <p>This can take longer than usual when the app has been idle.</p>
                  </div>
                )}
              </div>
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
                {isLoading ? (
                  <span className="dot-spinner">
                    <span></span>
                    <span></span>
                    <span></span>
                  </span>
                ) : (
                  <span id="upload-icon">⬆</span>
                )}
                <h2>{isLoading ? 'Running your point through CourtHawk' : 'Drop your point'}</h2>
                <p>{isLoading ? 'Breaking down every shot…' : 'Click or drag a video file to get started'}</p>
                {!isLoading && (
                  <button id="try-sample-button" onClick={handleTrySample}>
                    Try this sample
                  </button>
                )}
                {error && <p id="upload-error">{error}</p>}
              </div>
            )}
          </div>
        </section>
        <section id="stats">
          <AnalysisTimeline analysis={analysis} currentFrame={currentFrame} />
        </section>
      </div>
    </div>
  )
}

export default App
