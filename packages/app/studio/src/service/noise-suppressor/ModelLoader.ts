import {Errors, Procedure, Progress} from "@opendaw/lib-std"
import {
    DeepFilter3Variant,
    ModelAsset,
    allDeepFilter3Assets,
    getDeepFilter3Assets
} from "./ModelAssets"
import {readFromOpfs, removeFromOpfs, writeToOpfs} from "../ort-shared/OpfsCache"
import {fetchWithProgress} from "../ort-shared/FetchWithProgress"

export type DeepFilterBuffers = {
    readonly encoder: ArrayBuffer
    readonly erbDecoder: ArrayBuffer
    readonly dfDecoder: ArrayBuffer
}

export type ModelLoadCallbacks = {
    readonly onProgress: Progress.Handler
    readonly log: Procedure<string>
    readonly signal: AbortSignal
    readonly variant?: DeepFilter3Variant
}

const loadOneAsset = async (
    asset: ModelAsset,
    onProgress: Progress.Handler,
    log: Procedure<string>,
    signal: AbortSignal
): Promise<ArrayBuffer> => {
    if (signal.aborted) {throw Errors.AbortError}
    log(`Checking cache for ${asset.label}…`)
    const cached = await readFromOpfs(asset.opfsPath)
    if (cached !== null) {
        log(`✓ ${asset.label} loaded from cache (${(cached.byteLength / 1_048_576).toFixed(1)} MB)`)
        onProgress(1)
        return cached
    }
    log(`${asset.label} not cached — downloading…`)
    const url = new URL(asset.urlPath, document.baseURI).toString()
    const [fetchProgress, writeProgress] = Progress.splitWithWeights(onProgress, [95, 5])
    const buffer = await fetchWithProgress(url, fetchProgress, signal)
    if (signal.aborted) {throw Errors.AbortError}
    log(`Caching ${asset.label} (${(buffer.byteLength / 1_048_576).toFixed(1)} MB)…`)
    const wroteOk = await writeToOpfs(asset.opfsPath, buffer, error => {
        log(`OPFS write failed for ${asset.label}: ${Errors.toString(error)}`)
        removeFromOpfs(asset.opfsPath).catch(() => undefined)
    })
    writeProgress(1)
    if (wroteOk) {log(`✓ ${asset.label} cached`)}
    return buffer
}

export const loadDeepFilter3Models = async (
    {onProgress, log, signal, variant = "original"}: ModelLoadCallbacks
): Promise<DeepFilterBuffers> => {
    const assets = getDeepFilter3Assets(variant)
    const [encProgress, erbProgress, dfProgress] = Progress.splitWithWeights(onProgress, [
        assets.encoder.approxBytes,
        assets.erbDecoder.approxBytes,
        assets.dfDecoder.approxBytes
    ])
    const [encoder, erbDecoder, dfDecoder] = await Promise.all([
        loadOneAsset(assets.encoder, encProgress, log, signal),
        loadOneAsset(assets.erbDecoder, erbProgress, log, signal),
        loadOneAsset(assets.dfDecoder, dfProgress, log, signal)
    ])
    return {encoder, erbDecoder, dfDecoder}
}

export const clearDeepFilter3Cache = async (): Promise<void> => {
    await Promise.all(allDeepFilter3Assets.map(asset => removeFromOpfs(asset.opfsPath)))
}
