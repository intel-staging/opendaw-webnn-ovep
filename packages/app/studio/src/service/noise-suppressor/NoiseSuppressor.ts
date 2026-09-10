import {Errors, isDefined, Procedure, Progress} from "@opendaw/lib-std"
import type {InferenceSession} from "onnxruntime-web"
import {describeBackend, loadOrt, OrtBackend, pickPreferredBackend, probeBackends} from "../ort-shared/OrtRuntime"
export type {OrtBackend}
import {DeepFilter3Variant} from "./ModelAssets"
import {loadDeepFilter3Models} from "./ModelLoader"
import {resampleFrom48k, resampleTo48k} from "./pipeline/Resampler"
import {createAnalysisState, frameAnalysis} from "./pipeline/Analysis"
import {createSynthesisState, frameSynthesis} from "./pipeline/Synthesis"
import {computeErb, expandMask} from "./pipeline/ErbFilterbank"
import {createNormalizerState, normalizeErb, normalizeSpec} from "./pipeline/Normalizer"
import {applyDfFilter, createDfFilterState} from "./pipeline/DfFilter"
import {FRAME_SIZE, NB_DF, NB_ERB, NUM_FREQS} from "./pipeline/Constants"

export type NoiseSuppressorConfig = {
    readonly attenuationDb: number
}

export type DenoisedAudio = {
    readonly sampleRate: number
    readonly left: Float32Array
    readonly right: Float32Array
}

export type NoiseSuppressorCallbacks = {
    readonly onProgress: Progress.Handler
    readonly log: Procedure<string>
    readonly signal: AbortSignal
    readonly preferredBackend?: OrtBackend
}

type ModelBuffers = {
    readonly encoder: ArrayBuffer
    readonly erbDecoder: ArrayBuffer
    readonly dfDecoder: ArrayBuffer
}

type SessionPool = {
    readonly encoder: InferenceSession
    readonly erbDecoder: InferenceSession
    readonly dfDecoder: InferenceSession
}

const variantForBackend = (backend: OrtBackend): DeepFilter3Variant =>
    backend === "wasm" ? "original" : "webnn"

export class NoiseSuppressor {
    #modelBuffers: ModelBuffers | null = null
    #modelBufferVariant: DeepFilter3Variant | null = null
    #sessions: SessionPool | null = null
    #sessionNumFrames: number = -1
    #sessionBackend: OrtBackend | null = null
    #preferredBackend: OrtBackend = "wasm"
    #activeBackend: OrtBackend | null = null

    get activeBackend(): OrtBackend | null {return this.#activeBackend}
    get isModelLoaded(): boolean {return isDefined(this.#modelBuffers)}

    async ensureModelLoaded({onProgress, log, signal, preferredBackend}: NoiseSuppressorCallbacks): Promise<void> {
        const caps = await probeBackends()
        const backend = preferredBackend ?? pickPreferredBackend(caps)
        const variant = variantForBackend(backend)
        if (isDefined(this.#modelBuffers) && this.#preferredBackend === backend && this.#modelBufferVariant === variant) {
            onProgress(1)
            return
        }
        if (this.#preferredBackend !== backend) {
            if (isDefined(this.#sessions)) {
                log(`Switching device to ${describeBackend(backend)}…`)
                this.#disposeSessions()
            }
            this.#preferredBackend = backend
        }
        if (this.#modelBufferVariant !== variant) {
            this.#modelBuffers = null
            this.#modelBufferVariant = null
            if (isDefined(this.#sessions)) {this.#disposeSessions()}
        }
        if (isDefined(this.#modelBuffers)) {onProgress(1); return}
        const buffers = await loadDeepFilter3Models({onProgress, log, signal, variant})
        if (signal.aborted) {throw Errors.AbortError}
        this.#modelBuffers = buffers
        this.#modelBufferVariant = variant
    }

    async processFrames(
        name: string,
        frames: ReadonlyArray<Float32Array>,
        sampleRate: number,
        config: NoiseSuppressorConfig,
        callbacks: NoiseSuppressorCallbacks
    ): Promise<DenoisedAudio> {
        const {onProgress, log, signal} = callbacks
        if (!isDefined(this.#modelBuffers)) {
            throw new Error("Model not loaded — call ensureModelLoaded() first")
        }
        log(`Resampling ${name} to 48 kHz…`)
        const [resampleProgress, sessionProgress, inferProgress, resampleBackProgress] =
            Progress.splitWithWeights(onProgress, [5, 10, 80, 5])
        const frames48k = await resampleTo48k(frames, sampleRate)
        resampleProgress(1)
        if (signal.aborted) {throw Errors.AbortError}
        const numChannels = Math.min(frames48k.length, 2)
        const numFrames = Math.floor((frames48k[0].length - (960 - FRAME_SIZE)) / FRAME_SIZE) + 1
        await this.#getOrCreateSessions(numFrames, log)
        sessionProgress(1)
        if (signal.aborted) {throw Errors.AbortError}
        log(`Processing ${(frames48k[0].length / 48000).toFixed(1)}s × ${numChannels} channel(s)…`)
        const outputChannels: Float32Array[] = []
        for (let ch = 0; ch < numChannels; ch++) {
            if (signal.aborted) {throw Errors.AbortError}
            log(`  Channel ${ch + 1}/${numChannels}…`)
            const [chStart, chEnd] = Progress.splitWithWeights(inferProgress, [1, 1])
            const processed = await this.#processChannel(
                frames48k[ch], numFrames, config, log, signal,
                ch === 0 ? chStart : chEnd
            )
            outputChannels.push(processed)
        }
        inferProgress(1)
        if (signal.aborted) {throw Errors.AbortError}
        log(`Resampling back to ${sampleRate} Hz…`)
        const resampledBack = await resampleFrom48k(outputChannels, sampleRate)
        resampleBackProgress(1)
        const left = resampledBack[0]
        const right = numChannels === 2 ? resampledBack[1] : new Float32Array(left)
        return {sampleRate, left, right}
    }

    async #getOrCreateSessions(numFrames: number, log: Procedure<string>): Promise<void> {
        if (isDefined(this.#sessions) && this.#sessionNumFrames === numFrames && this.#sessionBackend === this.#preferredBackend) {return}
        if (isDefined(this.#sessions)) {this.#disposeSessions()}
        const buffers = this.#modelBuffers!
        const backend = this.#preferredBackend
        const cascade: OrtBackend[] = backend === "wasm" ? ["wasm"] : [backend, "wasm"]
        const ort = await loadOrt()
        const sharedMlContexts = new Map<OrtBackend, unknown>()
        const getMlContext = async (candidate: OrtBackend): Promise<unknown> => {
            const cached = sharedMlContexts.get(candidate)
            if (isDefined(cached)) {return cached}
            const deviceType = candidate === "webnn-npu" ? "npu" : "gpu"
            const webnnApi = (navigator as unknown as {ml: {createContext: (options: {deviceType: string}) => Promise<unknown>}}).ml
            const context = await webnnApi.createContext({deviceType})
            sharedMlContexts.set(candidate, context)
            return context
        }
        const createSession = async (buffer: ArrayBuffer, name: string): Promise<{session: InferenceSession, backend: OrtBackend}> => {
            for (const candidate of cascade) {
                const isWebnn = candidate === "webnn-npu" || candidate === "webnn-gpu"
                const options: InferenceSession.SessionOptions = {
                    executionProviders: [],
                    logSeverityLevel: isWebnn ? 0 : 3,
                    logVerbosityLevel: isWebnn ? 1 : 0,
                    ...(isWebnn ? {freeDimensionOverrides: {S: numFrames}} : {})
                }
                try {
                    if (isWebnn) {
                        const deviceType = candidate === "webnn-npu" ? "npu" : "gpu"
                        const mlContext = await getMlContext(candidate)
                        options.executionProviders = [{name: "webnn", deviceType, context: mlContext}]
                    } else {
                        options.executionProviders = [{name: "wasm"}]
                    }
                    log(`Creating ${name} on ${describeBackend(candidate)} (S=${numFrames})…`)
                    const session = await ort.InferenceSession.create(buffer, options)
                    log(`✓ ${name} ready on ${describeBackend(candidate)}`)
                    return {session, backend: candidate}
                } catch (error) {
                    console.error(`[NoiseSuppressor] ${name} on ${describeBackend(candidate)}`, error)
                    const message = error instanceof Error ? error.message : String(error)
                    log(`✗ ${name} failed on ${describeBackend(candidate)}: ${message}`)
                }
            }
            throw new Error(`All backends failed for ${name}`)
        }
        // Create sessions sequentially. Parallel `Promise.all` triggers a WebNN/OpenVINO race
        // that crashes ORT-Web's wasm runtime with "memory access out of bounds" during
        // session init when the graph has Scan nodes adjacent to WebNN-partitioned ops.
        const encResult = await createSession(buffers.encoder, "encoder")
        const erbResult = await createSession(buffers.erbDecoder, "ERB decoder")
        const dfResult = await createSession(buffers.dfDecoder, "DF decoder")
        this.#sessions = {encoder: encResult.session, erbDecoder: erbResult.session, dfDecoder: dfResult.session}
        this.#sessionNumFrames = numFrames
        this.#sessionBackend = this.#preferredBackend
        const activeBackends = new Set<OrtBackend>([encResult.backend, erbResult.backend, dfResult.backend])
        this.#activeBackend = activeBackends.size === 1 ? Array.from(activeBackends)[0] : "wasm"
        const summary = Array.from(activeBackends).map(describeBackend).join(" + ")
        log(`✓ Sessions ready (${summary})`)
    }

    async #processChannel(
        channelData: Float32Array,
        numFrames: number,
        config: NoiseSuppressorConfig,
        log: Procedure<string>,
        signal: AbortSignal,
        onProgress: Progress.Handler
    ): Promise<Float32Array> {
        const sessions = this.#sessions!
        const ort = await loadOrt()
        const analysisState = createAnalysisState()
        const normState = createNormalizerState()
        const allErbFeatures: Float32Array[] = []
        const allSpecFeatures: Float32Array[] = []
        const allComplexSpectra: Array<{real: Float32Array, imag: Float32Array}> = []
        const wnorm = 1.0 / ((960 * 960) / (2 * FRAME_SIZE))
        for (let t = 0; t < numFrames; t++) {
            const frameStart = t * FRAME_SIZE
            const frame = new Float32Array(FRAME_SIZE)
            const copyLen = Math.min(FRAME_SIZE, channelData.length - frameStart)
            if (copyLen > 0) {frame.set(channelData.subarray(frameStart, frameStart + copyLen))}
            const {real, imag} = frameAnalysis(frame, analysisState)
            const scaledReal = new Float32Array(NUM_FREQS)
            const scaledImag = new Float32Array(NUM_FREQS)
            for (let i = 0; i < NUM_FREQS; i++) {
                scaledReal[i] = real[i] * wnorm
                scaledImag[i] = imag[i] * wnorm
            }
            const magSq = new Float32Array(NUM_FREQS)
            for (let i = 0; i < NUM_FREQS; i++) {
                magSq[i] = scaledReal[i] * scaledReal[i] + scaledImag[i] * scaledImag[i]
            }
            const erbLinear = computeErb(magSq)
            const erbNorm = normalizeErb(erbLinear, normState)
            allErbFeatures.push(erbNorm)
            const specNorm = normalizeSpec(scaledReal.subarray(0, NB_DF), scaledImag.subarray(0, NB_DF), normState)
            allSpecFeatures.push(specNorm)
            allComplexSpectra.push({real: scaledReal, imag: scaledImag})
        }
        if (signal.aborted) {throw Errors.AbortError}
        const erbTensorData = new Float32Array(numFrames * NB_ERB)
        for (let t = 0; t < numFrames; t++) {
            erbTensorData.set(allErbFeatures[t], t * NB_ERB)
        }
        const specTensorData = new Float32Array(2 * numFrames * NB_DF)
        for (let t = 0; t < numFrames; t++) {
            const spec = allSpecFeatures[t]
            for (let i = 0; i < NB_DF; i++) {
                specTensorData[0 * numFrames * NB_DF + t * NB_DF + i] = spec[i * 2]
                specTensorData[1 * numFrames * NB_DF + t * NB_DF + i] = spec[i * 2 + 1]
            }
        }
        const featErb = new ort.Tensor("float32", erbTensorData, [1, 1, numFrames, NB_ERB])
        const featSpec = new ort.Tensor("float32", specTensorData, [1, 2, numFrames, NB_DF])
        log(`    Running encoder (${numFrames} frames)…`)
        const encOut = await sessions.encoder.run({feat_erb: featErb, feat_spec: featSpec})
        if (signal.aborted) {throw Errors.AbortError}
        const erbInputs: Record<string, unknown> = {emb: encOut.emb}
        for (const name of ["e0", "e1", "e2", "e3"] as const) {
            if (encOut[name]) {erbInputs[name] = encOut[name]}
        }
        const erbOut = await sessions.erbDecoder.run(erbInputs as Parameters<typeof sessions.erbDecoder.run>[0])
        if (signal.aborted) {throw Errors.AbortError}
        const dfInputs: Record<string, unknown> = {emb: encOut.emb}
        if (encOut.c0) {dfInputs.c0 = encOut.c0}
        const dfOut = await sessions.dfDecoder.run(dfInputs as Parameters<typeof sessions.dfDecoder.run>[0])
        if (signal.aborted) {throw Errors.AbortError}
        onProgress(0.9)
        const erbMaskData = (erbOut.m ?? erbOut.erb_mask).data as Float32Array
        const dfCoefsData = dfOut.coefs.data as Float32Array
        const attenuationGain = Math.pow(10, -config.attenuationDb / 20)
        const maskedSpectra: Array<{real: Float32Array, imag: Float32Array}> = []
        for (let t = 0; t < numFrames; t++) {
            const erbMask32 = new Float32Array(NB_ERB)
            for (let i = 0; i < NB_ERB; i++) {
                const val = erbMaskData[t * NB_ERB + i]
                const clamped = isFinite(val) ? Math.max(0, Math.min(1, val)) : 0.5
                erbMask32[i] = 1 - (1 - clamped) * attenuationGain
            }
            const fullMask = expandMask(erbMask32)
            const orig = allComplexSpectra[t]
            const masked = {real: new Float32Array(NUM_FREQS), imag: new Float32Array(NUM_FREQS)}
            for (let i = 0; i < NUM_FREQS; i++) {
                masked.real[i] = orig.real[i] * fullMask[i]
                masked.imag[i] = orig.imag[i] * fullMask[i]
            }
            maskedSpectra.push(masked)
        }
        const dfFilterState = createDfFilterState()
        const synthState = createSynthesisState(analysisState.fft)
        const outputLength = numFrames * FRAME_SIZE
        const outputSignal = new Float32Array(outputLength)
        for (let t = 0; t < numFrames; t++) {
            const orig = allComplexSpectra[t]
            const perFrameCoefs = new Float32Array(NB_DF * 10)
            for (let i = 0; i < NB_DF * 10; i++) {
                perFrameCoefs[i] = dfCoefsData[t * NB_DF * 10 + i]
            }
            const {real: dfReal, imag: dfImag} = applyDfFilter(orig.real, orig.imag, perFrameCoefs, dfFilterState)
            const masked = maskedSpectra[t]
            for (let i = NB_DF; i < NUM_FREQS; i++) {
                dfReal[i] = masked.real[i]
                dfImag[i] = masked.imag[i]
            }
            const beta = 0.02
            const eps = 1e-12
            for (let i = 0; i < NUM_FREQS; i++) {
                const origMag = Math.sqrt(orig.real[i] * orig.real[i] + orig.imag[i] * orig.imag[i])
                const enhMag = Math.sqrt(dfReal[i] * dfReal[i] + dfImag[i] * dfImag[i])
                const mask = Math.max(eps, Math.min(1, enhMag / (origMag + eps)))
                const maskSin = Math.max(eps, mask * Math.sin(Math.PI * mask / 2))
                const pf = (1 + beta) / (1 + beta * (mask / maskSin) * (mask / maskSin))
                dfReal[i] *= pf
                dfImag[i] *= pf
            }
            const frame = frameSynthesis(dfReal, dfImag, synthState)
            outputSignal.set(frame, t * FRAME_SIZE)
        }
        for (let i = 0; i < outputLength; i++) {
            const s = outputSignal[i]
            outputSignal[i] = isFinite(s) ? Math.max(-1, Math.min(1, s)) : 0
        }
        onProgress(1)
        return outputSignal
    }

    #disposeSessions(): void {
        if (isDefined(this.#sessions)) {
            const {encoder, erbDecoder, dfDecoder} = this.#sessions
            encoder.release?.().catch(() => undefined)
            erbDecoder.release?.().catch(() => undefined)
            dfDecoder.release?.().catch(() => undefined)
            this.#sessions = null
            this.#sessionNumFrames = -1
            this.#sessionBackend = null
            this.#activeBackend = null
        }
    }

    dispose(): void {
        this.#disposeSessions()
        this.#modelBuffers = null
    }
}
