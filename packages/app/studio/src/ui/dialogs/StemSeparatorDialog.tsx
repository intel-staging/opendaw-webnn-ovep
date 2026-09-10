import css from "./StemSeparatorDialog.sass?inline"
import {Html} from "@opendaw/lib-dom"
import {createElement} from "@opendaw/lib-jsx"
import {DefaultObservableValue, Errors, isDefined, Terminator} from "@opendaw/lib-std"
import {Colors, IconSymbol} from "@opendaw/studio-enums"
import {Button} from "@/ui/components/Button"
import {Dialog} from "@/ui/components/Dialog"
import {ProgressBar} from "@/ui/components/ProgressBar"
import {Surface} from "@/ui/surface/Surface"
import {StudioService} from "@/service/StudioService"
import {availableBackends, describeBackend, OrtBackend, probeBackends} from "@/service/stem-separator/OrtRuntime"
import {SeparatedStems, StemSeparator} from "@/service/stem-separator/StemSeparator"

const className = Html.adoptStyleSheet(css, "StemSeparatorDialog")

let sharedSeparator: StemSeparator | null = null
const getSharedSeparator = (): StemSeparator => {
    if (!isDefined(sharedSeparator)) {sharedSeparator = new StemSeparator()}
    return sharedSeparator
}

type DialogState = "idle" | "loading-model" | "ready" | "separating" | "done" | "error"

export type PreloadedAudio = {
    readonly name: string
    readonly frames: ReadonlyArray<Float32Array>
    readonly sampleRate: number
}

export const showStemSeparatorDialog = async (service: StudioService, preloaded?: PreloadedAudio): Promise<void> => {
    const {resolve, promise} = Promise.withResolvers<void>()
    const lifecycle = new Terminator()
    const separator = getSharedSeparator()
    let currentAbort: AbortController | null = null
    const abortCurrent = () => {
        if (isDefined(currentAbort)) {currentAbort.abort()}
    }
    const newOperation = (): AbortSignal => {
        currentAbort = new AbortController()
        return currentAbort.signal
    }
    lifecycle.own({terminate: () => abortCurrent()})
    let elapsedTimer: ReturnType<typeof setInterval> | null = null
    let separationStartTime = 0
    const startElapsedTimer = () => {
        separationStartTime = performance.now()
        elapsedTimer = setInterval(() => {
            const secs = Math.floor((performance.now() - separationStartTime) / 1000)
            const mm = String(Math.floor(secs / 60)).padStart(2, "0")
            const ss = String(secs % 60).padStart(2, "0")
            elapsedEl.textContent = `${mm}:${ss}`
        }, 1000)
    }
    const stopElapsedTimer = () => {
        if (isDefined(elapsedTimer)) {
            clearInterval(elapsedTimer)
            elapsedTimer = null
        }
    }
    lifecycle.own({terminate: () => stopElapsedTimer()})

    const state = new DefaultObservableValue<DialogState>(separator.isModelLoaded ? "ready" : "idle")
    const progress = new DefaultObservableValue<number>(0)
    const statusMessage = new DefaultObservableValue<string>(
        separator.isModelLoaded
            ? (isDefined(preloaded) ? `Ready to separate "${preloaded.name}".` : "Pick an audio file to separate.")
            : "Load the HTDemucs model to get started."
    )
    const logMessages = new DefaultObservableValue<ReadonlyArray<string>>([])
    const selectedFile = new DefaultObservableValue<File | null>(null)
    const selectedBackend = new DefaultObservableValue<OrtBackend>("wasm")

    const pushLog = (message: string) => {
        const lines = logMessages.getValue()
        logMessages.setValue([...lines.slice(-99), message])
    }

    const progressContainer: HTMLElement = (
        <div className="hidden"><ProgressBar lifecycle={lifecycle} progress={progress}/></div>
    )
    const statusLine: HTMLElement = <div className="status"/>
    const elapsedEl: HTMLElement = <span className="elapsed hidden"/>
    const backendBadge: HTMLElement = <span className="backend-badge hidden"/>
    const deviceSelect: HTMLSelectElement = <select className="device-select"/>
    const deviceRow: HTMLElement = <div className="device-row hidden"><label>Device</label>{deviceSelect}</div>
    const filePicker: HTMLInputElement = <input type="file" accept="audio/*" className="hidden"/>

    const populateDeviceSelector = async () => {
        const placeholder: HTMLOptionElement = <option value="" disabled selected>Detecting devices…</option>
        deviceSelect.appendChild(placeholder)
        deviceSelect.disabled = true
        const caps = await probeBackends()
        const backends = availableBackends(caps)
        deviceSelect.replaceChildren()
        deviceSelect.disabled = false
        for (const backend of backends) {
            const option: HTMLOptionElement = <option value={backend}>{describeBackend(backend)}</option>
            deviceSelect.appendChild(option)
        }
        selectedBackend.setValue(backends[0])
        deviceSelect.value = backends[0]
    }
    deviceSelect.addEventListener("change", () => {
        selectedBackend.setValue(deviceSelect.value as OrtBackend)
        if (separator.isModelLoaded) {void loadModel()}
    })
    void populateDeviceSelector()
    const fileDrop: HTMLElement = (
        <div className="file-drop hidden">
            <strong>Drop an audio file here</strong>
            <div>or click to browse (mp3, wav, flac, ogg, m4a…)</div>
        </div>
    )
    const fileSelected: HTMLElement = <div className="file-selected hidden"/>
    const logEl: HTMLElement = <div className="log"/>

    lifecycle.own(statusMessage.subscribe(owner => statusLine.textContent = owner.getValue()))
    statusLine.textContent = statusMessage.getValue()

    lifecycle.own(logMessages.subscribe(owner => {
        logEl.textContent = owner.getValue().join("\n")
        logEl.scrollTop = logEl.scrollHeight
    }))

    const updateBackendBadge = () => {
        const backend = separator.activeBackend
        if (isDefined(backend)) {
            backendBadge.textContent = describeBackend(backend)
            backendBadge.classList.remove("hidden")
        } else {
            backendBadge.classList.add("hidden")
        }
    }

    const renderSelectedFile = (file: File | null) => {
        if (!isDefined(file)) {
            fileSelected.classList.add("hidden")
            fileSelected.replaceChildren()
            return
        }
        fileSelected.classList.remove("hidden")
        const filenameEl: HTMLElement = <div className="filename">{file.name}</div>
        const metaEl: HTMLElement = (
            <div className="filemeta">{(file.size / 1_048_576).toFixed(1)} MB · {file.type || "unknown type"}</div>
        )
        fileSelected.replaceChildren(filenameEl, metaEl)
    }
    lifecycle.own(selectedFile.subscribe(owner => renderSelectedFile(owner.getValue())))

    filePicker.addEventListener("change", () => {
        const file = filePicker.files?.[0]
        if (isDefined(file)) {selectedFile.setValue(file)}
    })
    fileDrop.addEventListener("click", () => filePicker.click())
    fileDrop.addEventListener("dragover", event => {
        event.preventDefault()
        fileDrop.classList.add("drag-active")
    })
    fileDrop.addEventListener("dragleave", () => fileDrop.classList.remove("drag-active"))
    fileDrop.addEventListener("drop", event => {
        event.preventDefault()
        fileDrop.classList.remove("drag-active")
        const file = event.dataTransfer?.files?.[0]
        if (isDefined(file)) {selectedFile.setValue(file)}
    })

    const buttonCallbacks: Record<string, () => void> = {}
    const makeButton = (key: string, text: string, primary: boolean): HTMLElement => {
        const button = (
            <Button lifecycle={lifecycle}
                    onClick={() => buttonCallbacks[key]?.()}
                    appearance={primary
                        ? {framed: true, color: Colors.blue}
                        : {color: Colors.gray}}>
                <span>{text}</span>
            </Button>
        )
        button.classList.add("hidden")
        return button
    }
    const btnLoadModel = makeButton("loadModel", "Load Model", true)
    const btnSeparate = makeButton("separate", "Separate", true)
    const btnCancel = makeButton("cancel", "Cancel", false)
    const btnAbort = makeButton("abort", "Abort", false)
    const btnClose = makeButton("close", "Close", true)
    const btnSeparateAnother = makeButton("separateAnother", "Separate Another", false)

    const footerEl: HTMLElement = (
        <footer className="stem-footer">
            {btnCancel}
            {btnSeparateAnother}
            {btnAbort}
            {btnLoadModel}
            {btnSeparate}
            {btnClose}
        </footer>
    )

    const dialog: HTMLDialogElement = (
        <Dialog headline="Separate Stems (HTDemucs)"
                icon={IconSymbol.FileList}
                cancelable={true}
                onCancel={() => abortCurrent()}
                buttons={[]}>
            <div className={className}>
                <div className="status-row">{statusLine}{elapsedEl}</div>
                {backendBadge}
                {deviceRow}
                {progressContainer}
                {fileDrop}
                {fileSelected}
                {filePicker}
                {logEl}
                {footerEl}
            </div>
        </Dialog>
    )

    const setButtons = (visible: ReadonlySet<string>) => {
        const all: ReadonlyArray<[string, HTMLElement]> = [
            ["loadModel", btnLoadModel],
            ["separate", btnSeparate],
            ["cancel", btnCancel],
            ["abort", btnAbort],
            ["close", btnClose],
            ["separateAnother", btnSeparateAnother]
        ]
        for (const [key, el] of all) {
            if (visible.has(key)) {el.classList.remove("hidden")}
            else {el.classList.add("hidden")}
        }
    }

    const showProgress = (visible: boolean) => {
        if (visible) {progressContainer.classList.remove("hidden")}
        else {progressContainer.classList.add("hidden")}
    }

    buttonCallbacks.loadModel = () => {void loadModel()}
    buttonCallbacks.separate = () => {void runSeparation()}
    buttonCallbacks.cancel = () => dialog.close()
    buttonCallbacks.abort = () => abortCurrent()
    buttonCallbacks.close = () => dialog.close()
    buttonCallbacks.separateAnother = () => state.setValue("ready")

    const renderForState = (next: DialogState) => {
        const showDeviceRow = next === "idle" || next === "ready"
        if (showDeviceRow) {deviceRow.classList.remove("hidden")} else {deviceRow.classList.add("hidden")}
        const showElapsed = next === "separating"
        if (showElapsed) {elapsedEl.classList.remove("hidden")} else {elapsedEl.classList.add("hidden")}
        switch (next) {
            case "idle":
                statusMessage.setValue("Load the HTDemucs model to get started.")
                showProgress(false)
                fileDrop.classList.add("hidden")
                setButtons(new Set(["cancel", "loadModel"]))
                break
            case "loading-model":
                statusMessage.setValue("Loading model…")
                showProgress(true)
                fileDrop.classList.add("hidden")
                setButtons(new Set(["abort"]))
                break
            case "ready":
                if (isDefined(preloaded)) {
                    statusMessage.setValue(`Ready to separate "${preloaded.name}".`)
                    fileDrop.classList.add("hidden")
                } else {
                    statusMessage.setValue("Pick an audio file to separate.")
                    fileDrop.classList.remove("hidden")
                }
                showProgress(false)
                setButtons(new Set(["cancel", "separate"]))
                break
            case "separating":
                statusMessage.setValue("Separating…")
                showProgress(true)
                fileDrop.classList.add("hidden")
                setButtons(new Set(["abort"]))
                break
            case "done":
                statusMessage.setValue("Stems imported to the project.")
                showProgress(false)
                fileDrop.classList.remove("hidden")
                selectedFile.setValue(null)
                setButtons(new Set(["close", "separateAnother"]))
                break
            case "error":
                showProgress(false)
                setButtons(new Set(["close"]))
                break
        }
        updateBackendBadge()
    }
    lifecycle.own(state.subscribe(owner => renderForState(owner.getValue())))

    const loadModel = async () => {
        state.setValue("loading-model")
        progress.setValue(0)
        const signal = newOperation()
        try {
            await separator.ensureModelLoaded({
                onProgress: value => progress.setValue(value),
                log: pushLog,
                signal,
                preferredBackend: selectedBackend.getValue()
            })
            state.setValue("ready")
        } catch (error) {
            if (Errors.isAbort(error)) {
                pushLog("Aborted.")
                state.setValue("idle")
                return
            }
            pushLog(`Error loading model: ${Errors.toString(error)}`)
            statusMessage.setValue("Failed to load model. See log for details.")
            state.setValue("error")
        }
    }

    const runSeparation = async () => {
        const file = selectedFile.getValue()
        if (!isDefined(preloaded) && !isDefined(file)) {
            statusMessage.setValue("Please pick an audio file first.")
            return
        }
        state.setValue("separating")
        progress.setValue(0)
        elapsedEl.textContent = "00:00"
        startElapsedTimer()
        const signal = newOperation()
        try {
            const sourceName = isDefined(preloaded) ? preloaded.name : file!.name
            const stems = isDefined(preloaded)
                ? await separator.separateFrames(preloaded.name, preloaded.frames, preloaded.sampleRate, {
                    onProgress: value => progress.setValue(value),
                    log: pushLog,
                    signal
                })
                : await separator.separate(file!, {
                    onProgress: value => progress.setValue(value),
                    log: pushLog,
                    signal
                })
            stopElapsedTimer()
            pushLog("Importing stems into project…")
            await importToProject(service, sourceName, stems)
            pushLog("✓ Done.")
            state.setValue("done")
        } catch (error) {
            stopElapsedTimer()
            if (Errors.isAbort(error)) {
                pushLog("Aborted.")
                state.setValue("ready")
                return
            }
            pushLog(`Error separating: ${Errors.toString(error)}`)
            statusMessage.setValue("Separation failed. See log for details.")
            state.setValue("error")
        }
    }

    dialog.addEventListener("close", () => {
        lifecycle.terminate()
        resolve()
    })
    Surface.get().flyout.appendChild(dialog)
    dialog.showModal()
    renderForState(state.getValue())
    return promise
}

const importToProject = async (
    service: StudioService,
    sourceFilename: string,
    stems: SeparatedStems
): Promise<void> => {
    await service.importSeparatedStems(sourceFilename, stems.sampleRate, {
        drums: stems.drums,
        bass: stems.bass,
        other: stems.other,
        vocals: stems.vocals
    })
}
