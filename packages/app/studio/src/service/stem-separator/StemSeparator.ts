import {Errors, isDefined, Procedure, Progress} from "@opendaw/lib-std"
import type {InferenceSession} from "onnxruntime-web"
import {describeBackend, loadOrt, OrtBackend, pickPreferredBackend, probeBackends} from "./OrtRuntime"
export type {OrtBackend}
import {loadHtdemucsFwdModel} from "./ModelLoader"
import {decodeFileToStereo44100, resampleFramesToStereo44100} from "./pipeline/WavDecoder"
import {buildTriangularWeights, chunkStereo} from "./pipeline/Chunker"
import {htdemucsPreForward} from "./pipeline/PreForward"
import {htdemucsPostForward} from "./pipeline/PostForward"
import {StftContext} from "./pipeline/Stft"
import {
    NUM_STEMS,
    SEGMENT_LENGTH,
    STEM_NAMES,
    STFT_FREQ_BINS,
    STFT_TIME_FRAMES,
    TARGET_SAMPLE_RATE
} from "./pipeline/Constants"

export type SeparatorChannels = {
    readonly left: Float32Array
    readonly right: Float32Array
}

export type SeparatedStems = {
    readonly sampleRate: number
    readonly totalSamples: number
    readonly drums: SeparatorChannels
    readonly bass: SeparatorChannels
    readonly other: SeparatorChannels
    readonly vocals: SeparatorChannels
}

export type SeparatorCallbacks = {
    readonly onProgress: Progress.Handler
    readonly log: Procedure<string>
    readonly signal: AbortSignal
    readonly preferredBackend?: OrtBackend
}

export class StemSeparator {
    #session: InferenceSession | null = null
    #activeBackend: OrtBackend | null = null
    readonly #stft = new StftContext()

    get activeBackend(): OrtBackend | null {return this.#activeBackend}
    get isModelLoaded(): boolean {return isDefined(this.#session)}

    async ensureModelLoaded({onProgress, log, signal, preferredBackend}: SeparatorCallbacks): Promise<void> {
        const caps = await probeBackends()
        const backend = preferredBackend ?? pickPreferredBackend(caps)
        if (isDefined(this.#session) && this.#activeBackend === backend) {onProgress(1); return}
        if (isDefined(this.#session)) {
            log(`Switching device from ${describeBackend(this.#activeBackend!)} to ${describeBackend(backend)}…`)
            this.dispose()
        }
        const [fetchProgress, sessionProgress] = Progress.splitWithWeights(onProgress, [90, 10])
        const {graph, weights} = await loadHtdemucsFwdModel({onProgress: fetchProgress, log, signal})
        if (signal.aborted) {throw Errors.AbortError}
        log(`Creating ONNX session on ${describeBackend(backend)}…`)
        const ort = await loadOrt()
        const sessionOptions: InferenceSession.SessionOptions = {
            executionProviders: [],
            logSeverityLevel: 2,
            externalData: [{data: weights, path: "htdemucs_fwd.onnx.data"}]
        }
        if (backend === "webnn-npu" || backend === "webnn-gpu") {
            const deviceType = backend === "webnn-npu" ? "npu" : "gpu"
            const webnnApi = (navigator as unknown as {ml: {createContext: (options: {deviceType: string}) => Promise<unknown>}}).ml
            const mlContext = await webnnApi.createContext({deviceType})
            sessionOptions.executionProviders = [{name: "webnn", deviceType, context: mlContext}]
        } else {
            sessionOptions.executionProviders = [{name: "wasm"}]
        }
        this.#session = await ort.InferenceSession.create(graph, sessionOptions)
        this.#activeBackend = backend
        sessionProgress(1)
        log(`✓ Session ready on ${describeBackend(backend)}`)
    }

    async separate(file: File, callbacks: SeparatorCallbacks): Promise<SeparatedStems> {
        const {onProgress, log, signal} = callbacks
        if (!isDefined(this.#session)) {
            throw new Error("Model not loaded — call ensureModelLoaded() first")
        }
        const [decodeProgress, inferenceProgress] = Progress.splitWithWeights(onProgress, [5, 95])
        log(`Decoding ${file.name}…`)
        const decoded = await decodeFileToStereo44100(file)
        decodeProgress(1)
        if (signal.aborted) {throw Errors.AbortError}
        log(`Decoded ${decoded.left.length.toLocaleString()} samples @ ${TARGET_SAMPLE_RATE} Hz`)
        return this.#runInference(decoded.left, decoded.right, inferenceProgress, log, signal)
    }

    async separateFrames(
        name: string,
        frames: ReadonlyArray<Float32Array>,
        sampleRate: number,
        callbacks: SeparatorCallbacks
    ): Promise<SeparatedStems> {
        const {onProgress, log, signal} = callbacks
        if (!isDefined(this.#session)) {
            throw new Error("Model not loaded — call ensureModelLoaded() first")
        }
        const [resampleProgress, inferenceProgress] = Progress.splitWithWeights(onProgress, [3, 97])
        log(`Resampling ${name}…`)
        const {left, right} = await resampleFramesToStereo44100(frames, sampleRate)
        resampleProgress(1)
        if (signal.aborted) {throw Errors.AbortError}
        log(`Resampled to ${left.length.toLocaleString()} samples @ ${TARGET_SAMPLE_RATE} Hz`)
        return this.#runInference(left, right, inferenceProgress, log, signal)
    }

    async #runInference(
        left: Float32Array,
        right: Float32Array,
        onProgress: Progress.Handler,
        log: Procedure<string>,
        signal: AbortSignal
    ): Promise<SeparatedStems> {
        const session = this.#session
        if (!isDefined(session)) {throw new Error("Session missing")}
        const ort = await loadOrt()
        const {segments, totalSamples} = chunkStereo(left, right)
        log(`Separating ${segments.length} segments…`)
        const weights = buildTriangularWeights()
        const stems: Array<[Float32Array, Float32Array]> = []
        for (let stem = 0; stem < NUM_STEMS; stem++) {
            stems.push([new Float32Array(totalSamples), new Float32Array(totalSamples)])
        }
        const weightSum = new Float32Array(totalSamples)
        const startTime = performance.now()
        for (let segIdx = 0; segIdx < segments.length; segIdx++) {
            if (signal.aborted) {throw Errors.AbortError}
            const seg = segments[segIdx]
            const segStart = performance.now()
            const pre = htdemucsPreForward(this.#stft, seg.data)
            const xTensor = new ort.Tensor("float32", pre.xBuf, [1, 4, STFT_FREQ_BINS, STFT_TIME_FRAMES])
            const xtTensor = new ort.Tensor("float32", pre.xtBuf, [1, 2, SEGMENT_LENGTH])
            const output = await session.run({x: xTensor, xt: xtTensor})
            if (signal.aborted) {throw Errors.AbortError}
            const xOut = output.x_out
            const xtOut = output.xt_out
            const sources = htdemucsPostForward(
                this.#stft,
                xOut.data as Float32Array,
                xtOut.data as Float32Array,
                pre
            )
            const writeLimit = Math.min(SEGMENT_LENGTH, totalSamples - seg.startSample)
            for (let stem = 0; stem < NUM_STEMS; stem++) {
                const stemL = stems[stem][0]
                const stemR = stems[stem][1]
                const offsetL = stem * 2 * SEGMENT_LENGTH
                const offsetR = offsetL + SEGMENT_LENGTH
                for (let k = 0; k < writeLimit; k++) {
                    const weight = weights[k]
                    const destIdx = seg.startSample + k
                    stemL[destIdx] += sources[offsetL + k] * weight
                    stemR[destIdx] += sources[offsetR + k] * weight
                }
            }
            for (let k = 0; k < writeLimit; k++) {
                weightSum[seg.startSample + k] += weights[k]
            }
            xTensor.dispose?.()
            xtTensor.dispose?.()
            xOut.dispose?.()
            xtOut.dispose?.()
            const segMs = performance.now() - segStart
            log(`  segment ${segIdx + 1}/${segments.length} (${(segMs / 1000).toFixed(2)}s)`)
            onProgress((segIdx + 1) / segments.length)
        }
        for (let stem = 0; stem < NUM_STEMS; stem++) {
            for (let chan = 0; chan < 2; chan++) {
                const buffer = stems[stem][chan]
                for (let i = 0; i < totalSamples; i++) {
                    const wSum = weightSum[i]
                    if (wSum > 0) {buffer[i] /= wSum}
                }
            }
        }
        const totalSeconds = (performance.now() - startTime) / 1000
        const audioDuration = totalSamples / TARGET_SAMPLE_RATE
        log(`✓ ${totalSeconds.toFixed(1)}s for ${audioDuration.toFixed(1)}s of audio (${(audioDuration / totalSeconds).toFixed(2)}× realtime)`)
        return {
            sampleRate: TARGET_SAMPLE_RATE,
            totalSamples,
            drums: {left: stems[0][0], right: stems[0][1]},
            bass: {left: stems[1][0], right: stems[1][1]},
            other: {left: stems[2][0], right: stems[2][1]},
            vocals: {left: stems[3][0], right: stems[3][1]}
        }
    }

    dispose(): void {
        if (isDefined(this.#session)) {
            this.#session.release?.().catch(() => undefined)
            this.#session = null
            this.#activeBackend = null
        }
    }
}

export {STEM_NAMES}
