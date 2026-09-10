import {isDefined} from "@opendaw/lib-std"

export type OrtBackend = "webnn-npu" | "webnn-gpu" | "wasm"

export type OrtBackendCapabilities = {
    readonly webnnNpu: boolean
    readonly webnnGpu: boolean
}

type OrtModule = typeof import("onnxruntime-web")

let ortPromise: Promise<OrtModule> | null = null

export const loadOrt = (): Promise<OrtModule> => {
    if (!isDefined(ortPromise)) {
        // The all-EPs bundle, not the default entry: the default's jsep wasm hits an
        // "out of bounds" memory access during session init when a graph has `Scan` nodes
        // adjacent to WebNN-partitioned nodes, which is exactly our rewritten DF3 models.
        ortPromise = import("onnxruntime-web/all").then(ort => {
            ort.env.wasm.numThreads = navigator.hardwareConcurrency ?? 4
            ort.env.wasm.simd = true
            ort.env.logLevel = "warning"
            return ort as unknown as OrtModule
        })
    }
    return ortPromise
}

export const probeBackends = async (): Promise<OrtBackendCapabilities> => {
    const webnnApi = (navigator as unknown as {ml?: {createContext: (options: {deviceType: string}) => Promise<unknown>}}).ml
    if (!isDefined(webnnApi)) {return {webnnNpu: false, webnnGpu: false}}
    let webnnNpu = false
    let webnnGpu = false
    try {
        await webnnApi.createContext({deviceType: "npu"})
        webnnNpu = true
    } catch {
        /* unavailable */
    }
    try {
        await webnnApi.createContext({deviceType: "gpu"})
        webnnGpu = true
    } catch {
        /* unavailable */
    }
    return {webnnNpu, webnnGpu}
}

export const availableBackends = ({webnnNpu, webnnGpu}: OrtBackendCapabilities): ReadonlyArray<OrtBackend> => {
    const result: Array<OrtBackend> = []
    if (webnnNpu) {result.push("webnn-npu")}
    if (webnnGpu) {result.push("webnn-gpu")}
    result.push("wasm")
    return result
}

export const pickPreferredBackend = ({webnnNpu, webnnGpu}: OrtBackendCapabilities): OrtBackend => {
    if (webnnNpu) {return "webnn-npu"}
    if (webnnGpu) {return "webnn-gpu"}
    return "wasm"
}

export const describeBackend = (backend: OrtBackend): string => {
    switch (backend) {
        case "webnn-npu": return "WebNN-NPU"
        case "webnn-gpu": return "WebNN-GPU"
        case "wasm": return "WASM"
    }
}
