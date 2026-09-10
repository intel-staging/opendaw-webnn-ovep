import {DF_OFFSET, DF_ORDER, NB_DF, NUM_FREQS} from "./Constants"

export type DfFilterState = {
    specBuffer: Array<{real: Float32Array, imag: Float32Array}>
    frameIdx: number
}

export const createDfFilterState = (): DfFilterState => {
    const specBuffer: Array<{real: Float32Array, imag: Float32Array}> = []
    for (let i = 0; i < DF_ORDER; i++) {
        specBuffer.push({real: new Float32Array(NUM_FREQS), imag: new Float32Array(NUM_FREQS)})
    }
    return {specBuffer, frameIdx: 0}
}

export const applyDfFilter = (
    specReal: Float32Array,
    specImag: Float32Array,
    coefs: Float32Array,
    state: DfFilterState
): {real: Float32Array, imag: Float32Array} => {
    const slot = state.frameIdx % DF_ORDER
    state.specBuffer[slot].real.set(specReal)
    state.specBuffer[slot].imag.set(specImag)
    const outReal = new Float32Array(NUM_FREQS)
    const outImag = new Float32Array(NUM_FREQS)
    for (let f = 0; f < NB_DF; f++) {
        let sumRe = 0, sumIm = 0
        for (let tap = 0; tap < DF_ORDER; tap++) {
            const bufIdx = ((state.frameIdx - DF_OFFSET + tap) % DF_ORDER + DF_ORDER) % DF_ORDER
            const sRe = state.specBuffer[bufIdx].real[f]
            const sIm = state.specBuffer[bufIdx].imag[f]
            const cRe = coefs[f * DF_ORDER * 2 + tap * 2]
            const cIm = coefs[f * DF_ORDER * 2 + tap * 2 + 1]
            sumRe += sRe * cRe - sIm * cIm
            sumIm += sRe * cIm + sIm * cRe
        }
        outReal[f] = sumRe
        outImag[f] = sumIm
    }
    for (let f = NB_DF; f < NUM_FREQS; f++) {
        outReal[f] = specReal[f]
        outImag[f] = specImag[f]
    }
    state.frameIdx++
    return {real: outReal, imag: outImag}
}
