import {Errors, isDefined, Procedure, Progress} from "@opendaw/lib-std"
import {ModelAsset, HtdemucsFwdGraph, HtdemucsFwdWeights, ModelAssetWeights} from "./ModelAssets"
import {readFromOpfs, removeFromOpfs, writeToOpfs} from "./OpfsCache"
import {fetchWithProgress} from "./FetchWithProgress"

export type ModelBuffers = {
    readonly graph: ArrayBuffer
    readonly weights: ArrayBuffer
}

export type ModelLoadCallbacks = {
    readonly onProgress: Progress.Handler
    readonly log: Procedure<string>
    readonly signal: AbortSignal
}

const isTruncated = (asset: ModelAsset, byteLength: number): boolean =>
    isDefined(asset.exactBytes) && byteLength !== asset.exactBytes

const loadOneAsset = async (
    asset: ModelAsset,
    onProgress: Progress.Handler,
    log: Procedure<string>,
    signal: AbortSignal
): Promise<ArrayBuffer> => {
    if (signal.aborted) {throw Errors.AbortError}
    log(`Checking cache for ${asset.label}…`)
    const cached = await readFromOpfs(asset.opfsPath)
    if (cached !== null && isTruncated(asset, cached.byteLength)) {
        // An interrupted write leaves a short file behind; reading it back yields a corrupt model.
        log(`✗ ${asset.label} cache is ${cached.byteLength} bytes, expected ${asset.exactBytes} — re-downloading`)
        await removeFromOpfs(asset.opfsPath)
    } else if (cached !== null) {
        log(`✓ ${asset.label} loaded from cache (${(cached.byteLength / 1_048_576).toFixed(1)} MB)`)
        onProgress(1)
        return cached
    }
    const url = new URL(asset.urlPath, document.baseURI).toString()
    log(`${asset.label} not cached — downloading ${url}`)
    const [fetchProgress, writeProgress] = Progress.splitWithWeights(onProgress, [95, 5])
    const buffer = await fetchWithProgress(url, fetchProgress, signal)
    if (signal.aborted) {throw Errors.AbortError}
    if (isTruncated(asset, buffer.byteLength)) {
        throw new Error(`${asset.label}: downloaded ${buffer.byteLength} bytes, expected ${asset.exactBytes}`)
    }
    log(`Caching ${asset.label} (${(buffer.byteLength / 1_048_576).toFixed(1)} MB)…`)
    const wroteOk = await writeToOpfs(asset.opfsPath, buffer, error => {
        log(`OPFS write failed for ${asset.label}: ${Errors.toString(error)}`)
        removeFromOpfs(asset.opfsPath).catch(() => undefined)
    })
    writeProgress(1)
    if (wroteOk) {log(`✓ ${asset.label} cached`)}
    return buffer
}

export const loadHtdemucsFwdModel = async ({onProgress, log, signal}: ModelLoadCallbacks): Promise<ModelBuffers> => {
    const [graphProgress, weightsProgress] = Progress.splitWithWeights(onProgress, ModelAssetWeights as Array<number>)
    const graph = await loadOneAsset(HtdemucsFwdGraph, graphProgress, log, signal)
    const weights = await loadOneAsset(HtdemucsFwdWeights, weightsProgress, log, signal)
    return {graph, weights}
}

export const clearHtdemucsFwdCache = async (): Promise<void> => {
    await removeFromOpfs(HtdemucsFwdGraph.opfsPath)
    await removeFromOpfs(HtdemucsFwdWeights.opfsPath)
}
