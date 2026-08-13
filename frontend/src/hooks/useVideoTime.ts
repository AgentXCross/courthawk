// Takes a reference to a <video> element and continously give back its current playback time.

import { useEffect, useState } from 'react'
import type { RefObject } from 'react'

// Tracks the <video> element's current player time in seconds as React state.
// isReady should flip from false to true once the <video> element this ref points to
// actually exists in the DOM (e.g. once analysis data arrives and VideoPlayer mounts it) —
// videoRef's identity never changes, so without this the effect below would only ever
// run once, before the <video> element exists, and never notice it show up later.
//
// Polls with requestAnimationFrame instead of listening for the native 'timeupdate' event —
// timeupdate only fires a few times a second (fine for a scrubber bar, not for smooth
// per-frame tracking), which is what made the mini-court dots jump instead of glide.
export function useVideoTime(videoRef: RefObject<HTMLVideoElement | null>, isReady: boolean): number {
    const [currentTime, setCurrentTime] = useState(0)

    useEffect(() => {
        const video = videoRef.current
        if (!video) return

        let frameId: number

        const tick = () => {
            const time = video.currentTime
            setCurrentTime((prev) => (prev === time ? prev : time))
            frameId = requestAnimationFrame(tick)
        }

        frameId = requestAnimationFrame(tick)
        return () => cancelAnimationFrame(frameId)
    }, [videoRef, isReady])

    return currentTime
}
