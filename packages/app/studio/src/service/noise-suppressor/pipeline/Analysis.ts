import {BluesteinFft} from "./BluesteinFft"
import {FRAME_SIZE, FFT_SIZE, NUM_FREQS} from "./Constants"
import {getVorbisWindow} from "./VorbisWindow"

export type AnalysisState = {
    mem: Float32Array
    fft: BluesteinFft
}

export const createAnalysisState = (): AnalysisState => ({
    mem: new Float32Array(FRAME_SIZE),
    fft: new BluesteinFft()
})

export const frameAnalysis = (
    frame: Float32Array,
    state: AnalysisState
): {real: Float32Array, imag: Float32Array} => {
    const win = getVorbisWindow()
    const buf = new Float32Array(FFT_SIZE)
    buf.set(state.mem)
    buf.set(frame, FRAME_SIZE)
    state.mem.set(frame)
    for (let i = 0; i < FFT_SIZE; i++) {
        buf[i] *= win[i]
    }
    const {real, imag} = state.fft.forward(buf)
    return {
        real: real.subarray(0, NUM_FREQS),
        imag: imag.subarray(0, NUM_FREQS)
    }
}
