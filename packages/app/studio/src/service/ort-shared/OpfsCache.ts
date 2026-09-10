import {isDefined, Procedure} from "@opendaw/lib-std"

const splitPath = (path: string): ReadonlyArray<string> => path.split("/").filter(segment => segment.length > 0)

const resolveDirectory = async (
    path: ReadonlyArray<string>,
    options: {create: boolean}
): Promise<FileSystemDirectoryHandle> => {
    let current = await navigator.storage.getDirectory()
    for (const segment of path) {
        current = await current.getDirectoryHandle(segment, {create: options.create})
    }
    return current
}

export const readFromOpfs = async (path: string): Promise<ArrayBuffer | null> => {
    const parts = splitPath(path)
    if (parts.length === 0) {return null}
    const fileName = parts[parts.length - 1]
    const directorySegments = parts.slice(0, -1)
    try {
        const directory = await resolveDirectory(directorySegments, {create: false})
        const fileHandle = await directory.getFileHandle(fileName)
        const file = await fileHandle.getFile()
        return await file.arrayBuffer()
    } catch {
        return null
    }
}

export const writeToOpfs = async (
    path: string,
    arrayBuffer: ArrayBuffer,
    onError?: Procedure<unknown>
): Promise<boolean> => {
    const parts = splitPath(path)
    if (parts.length === 0) {return false}
    const fileName = parts[parts.length - 1]
    const directorySegments = parts.slice(0, -1)
    try {
        const directory = await resolveDirectory(directorySegments, {create: true})
        const fileHandle = await directory.getFileHandle(fileName, {create: true})
        const writable = await fileHandle.createWritable()
        await writable.write(arrayBuffer)
        await writable.close()
        return true
    } catch (error) {
        if (isDefined(onError)) {onError(error)}
        return false
    }
}

export const removeFromOpfs = async (path: string): Promise<void> => {
    const parts = splitPath(path)
    if (parts.length === 0) {return}
    const fileName = parts[parts.length - 1]
    const directorySegments = parts.slice(0, -1)
    try {
        const directory = await resolveDirectory(directorySegments, {create: false})
        await directory.removeEntry(fileName)
    } catch {
        /* already gone */
    }
}
