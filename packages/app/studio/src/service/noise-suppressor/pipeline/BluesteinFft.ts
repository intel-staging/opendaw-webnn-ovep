import {FFT_SIZE} from "./Constants"

const radix2Fft = (real: Float64Array, imag: Float64Array, invert: boolean): void => {
    const n = real.length
    for (let i = 1, j = 0; i < n; i++) {
        let bit = n >> 1
        for (; j & bit; bit >>= 1) {j ^= bit}
        j ^= bit
        if (i < j) {
            let t = real[i]; real[i] = real[j]; real[j] = t
            t = imag[i]; imag[i] = imag[j]; imag[j] = t
        }
    }
    for (let len = 2; len <= n; len <<= 1) {
        const ang = (invert ? 2 : -2) * Math.PI / len
        const wRe = Math.cos(ang)
        const wIm = Math.sin(ang)
        for (let i = 0; i < n; i += len) {
            let curRe = 1.0, curIm = 0.0
            for (let j = 0; j < len / 2; j++) {
                const uRe = real[i + j]
                const uIm = imag[i + j]
                const vRe = real[i + j + len / 2] * curRe - imag[i + j + len / 2] * curIm
                const vIm = real[i + j + len / 2] * curIm + imag[i + j + len / 2] * curRe
                real[i + j] = uRe + vRe
                imag[i + j] = uIm + vIm
                real[i + j + len / 2] = uRe - vRe
                imag[i + j + len / 2] = uIm - vIm
                const nextRe = curRe * wRe - curIm * wIm
                curIm = curRe * wIm + curIm * wRe
                curRe = nextRe
            }
        }
    }
    if (invert) {
        for (let i = 0; i < n; i++) {real[i] /= n; imag[i] /= n}
    }
}

export class BluesteinFft {
    readonly #n: number
    readonly #m: number
    readonly #chirpRe: Float64Array
    readonly #chirpIm: Float64Array
    readonly #bRe: Float64Array
    readonly #bIm: Float64Array

    constructor(n: number = FFT_SIZE) {
        this.#n = n
        let m = 1
        while (m < 2 * n - 1) {m <<= 1}
        this.#m = m
        this.#chirpRe = new Float64Array(n)
        this.#chirpIm = new Float64Array(n)
        for (let i = 0; i < n; i++) {
            const phase = -Math.PI * i * i / n
            this.#chirpRe[i] = Math.cos(phase)
            this.#chirpIm[i] = Math.sin(phase)
        }
        this.#bRe = new Float64Array(m)
        this.#bIm = new Float64Array(m)
        for (let i = 0; i < n; i++) {
            this.#bRe[i] = this.#chirpRe[i]
            this.#bIm[i] = -this.#chirpIm[i]
        }
        for (let i = m - n + 1; i < m; i++) {
            const j = m - i
            this.#bRe[i] = this.#chirpRe[j]
            this.#bIm[i] = -this.#chirpIm[j]
        }
        radix2Fft(this.#bRe, this.#bIm, false)
    }

    forward(input: Float32Array): {real: Float32Array, imag: Float32Array} {
        const n = this.#n
        const m = this.#m
        const aRe = new Float64Array(m)
        const aIm = new Float64Array(m)
        for (let i = 0; i < n; i++) {
            aRe[i] = input[i] * this.#chirpRe[i]
            aIm[i] = input[i] * this.#chirpIm[i]
        }
        radix2Fft(aRe, aIm, false)
        for (let i = 0; i < m; i++) {
            const re = aRe[i] * this.#bRe[i] - aIm[i] * this.#bIm[i]
            const im = aRe[i] * this.#bIm[i] + aIm[i] * this.#bRe[i]
            aRe[i] = re
            aIm[i] = im
        }
        radix2Fft(aRe, aIm, true)
        const outReal = new Float32Array(n)
        const outImag = new Float32Array(n)
        for (let i = 0; i < n; i++) {
            outReal[i] = aRe[i] * this.#chirpRe[i] - aIm[i] * this.#chirpIm[i]
            outImag[i] = aRe[i] * this.#chirpIm[i] + aIm[i] * this.#chirpRe[i]
        }
        return {real: outReal, imag: outImag}
    }

    inverse(real: Float32Array, imag: Float32Array): Float32Array {
        // Compute inverse DFT via forward Bluestein: IFFT(X) = conj(FFT(conj(X))) / n.
        // Done this way to avoid bookkeeping around inverse-direction chirp/kernel conjugation
        // that the original implementation got wrong.
        const n = this.#n
        const conjInput = new Float32Array(n)
        const conjInputImag = new Float32Array(n)
        for (let i = 0; i < n; i++) {
            conjInput[i] = real[i]
            conjInputImag[i] = -imag[i]
        }
        const m = this.#m
        const aRe = new Float64Array(m)
        const aIm = new Float64Array(m)
        for (let i = 0; i < n; i++) {
            aRe[i] = conjInput[i] * this.#chirpRe[i] - conjInputImag[i] * this.#chirpIm[i]
            aIm[i] = conjInput[i] * this.#chirpIm[i] + conjInputImag[i] * this.#chirpRe[i]
        }
        radix2Fft(aRe, aIm, false)
        for (let i = 0; i < m; i++) {
            const re = aRe[i] * this.#bRe[i] - aIm[i] * this.#bIm[i]
            const im = aRe[i] * this.#bIm[i] + aIm[i] * this.#bRe[i]
            aRe[i] = re
            aIm[i] = im
        }
        radix2Fft(aRe, aIm, true)
        const out = new Float32Array(n)
        const scale = 1.0 / n
        for (let i = 0; i < n; i++) {
            // conj of forward output; return only real part (inverse of real input is real).
            const re = aRe[i] * this.#chirpRe[i] - aIm[i] * this.#chirpIm[i]
            out[i] = re * scale
        }
        return out
    }
}
