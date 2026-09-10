import {WINDOW_SIZE} from "./Constants"

let cachedWindow: Float32Array | null = null

export const getVorbisWindow = (): Float32Array => {
    if (cachedWindow !== null) {return cachedWindow}
    const win = new Float32Array(WINDOW_SIZE)
    for (let n = 0; n < WINDOW_SIZE; n++) {
        const t = Math.sin(Math.PI * (n + 0.5) / WINDOW_SIZE)
        win[n] = Math.sin(Math.PI / 2 * t * t)
    }
    cachedWindow = win
    return win
}
