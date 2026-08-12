// Contains the fronend function that sends a video to the FastAPI backend and gets the 
// analysis results back.

import type { PointAnalysis } from "../types/analysis";

// Move to an env var once ready to deploy, this is the backend address
const API_BASE_URL = 'http://localhost:8000'


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