// Contains the fronend function that sends a video to the FastAPI backend and gets the 
// analysis results back.

import type { PointAnalysis } from "../types/analysis";
import { API_BASE_URL } from "./config"


// Function is async so it is allowed to pause while it waits for something, 
// without stopping the rest of the program.
export async function analyzeVideo(file: File): Promise<PointAnalysis> {
    const formData = new FormData()
    formData.append('file', file) // 'file' must match the FastAPI route's parameter name exactly

    const response = await fetch(`${API_BASE_URL}/analyze`, {
        method: 'POST',
        body: formData,
    })

    if (!response.ok) {
        throw new Error(`Analysis failed: ${response.status} ${response.statusText}`)
    }

    return response.json() as Promise<PointAnalysis>
}

// Loads the pre-computed PointAnalysis for the sample video, bundled as a static file in
// frontend/public/. Instant, unlike analyzeVideo(), no pipeline run, just a JSON fetch.
// Regenerate this file with backend/generate_sample_analysis.py if the sample video or the
// pipeline itself changes.
export async function getSampleAnalysis(): Promise<PointAnalysis> {
    const response = await fetch('/sample-analysis.json')

    if (!response.ok) {
        throw new Error(`Failed to load sample analysis: ${response.status} ${response.statusText}`)
    }

    return response.json() as Promise<PointAnalysis>
}