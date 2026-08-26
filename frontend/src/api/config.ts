// Set VITE_API_BASE_URL in Vercel's project settings to the deployed Render backend's URL.
// Falls back to localhost so local dev (npm run dev) needs no .env file.
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'