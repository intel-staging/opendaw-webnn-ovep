#!/usr/bin/env node
// Fetches the ONNX models for the AI audio features into
// packages/app/studio/public/models/ and derives the DeepFilterNet3 WebNN
// variants locally. Run via `npm run models:fetch`.
import {createHash} from "node:crypto"
import {spawnSync} from "node:child_process"
import {existsSync, mkdirSync, mkdtempSync, readdirSync, renameSync, rmSync, statSync} from "node:fs"
import {tmpdir} from "node:os"
import {dirname, join, resolve} from "node:path"
import {fileURLToPath} from "node:url"

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..")
const MODELS_DIR = join(REPO_ROOT, "packages", "app", "studio", "public", "models")
const HTDEMUCS_DIR = join(MODELS_DIR, "htdemucs_fwd")
const DF3_DIR = join(MODELS_DIR, "df3")

const HTDEMUCS_BASE = "https://huggingface.co/Intel/demucs-openvino/resolve/main/htdemucs_v4"
const DF3_TARBALL = "https://raw.githubusercontent.com/Rikorose/DeepFilterNet/main/models/DeepFilterNet3_onnx.tar.gz"

const HTDEMUCS_FILES = [
    {name: "htdemucs_fwd.onnx", bytes: 2_385_507},
    {name: "htdemucs_fwd.onnx.data", bytes: 168_361_984}
]

// The tarball nests its members under tmp/export/.
const DF3_FILES = [
    {name: "enc.onnx", bytes: 1_954_042, sha256: "7c5399d3da8a50ebef1c1a0ae421b33376aa5e45d0e92df16da7e83c9c131916"},
    {name: "erb_dec.onnx", bytes: 3_292_397, sha256: "ab669a1d10afe20911728b33053a452071042317a90581092b325da7b2f9d895"},
    {name: "df_dec.onnx", bytes: 3_340_803, sha256: "23114ce3b0f6464b763ee62f7bb8aab6b2a129a21eabd5bcfe59413db05f278a"}
]

const force = process.argv.includes("--force")
const runVerify = process.argv.includes("--verify")

const fail = (message) => {
    console.error(`\n[error] ${message}\n`)
    process.exit(1)
}

const statSyncSafe = (path) => existsSync(path) ? statSync(path) : null

const sha256Of = async (path) => {
    const {readFile} = await import("node:fs/promises")
    return createHash("sha256").update(await readFile(path)).digest("hex")
}

const resolvePython = () => {
    for (const candidate of ["python3", "python"]) {
        const probe = spawnSync(candidate, ["-c", "import onnx, numpy"], {stdio: "pipe"})
        if (probe.status === 0) {return candidate}
    }
    return null
}

// curl rather than fetch(): it honours HTTP(S)_PROXY, which undici ignores.
const curl = (url, partial, extraArgs) =>
    spawnSync("curl", ["-fL", "--progress-bar", ...extraArgs, "-o", partial, url], {stdio: "inherit"})

const download = (url, destination) => {
    const partial = `${destination}.partial`
    console.log(`  ${url}`)
    let result = curl(url, partial, ["--retry", "3", "--retry-delay", "2"])
    // Intercepting proxies fail in assorted ways (35 = CRL unreachable, 52 = empty reply).
    if (result.status !== 0) {
        console.log(`  [retry] curl exited with ${result.status}, retrying with --ssl-no-revoke`)
        result = curl(url, partial, ["--ssl-no-revoke", "--retry", "3", "--retry-delay", "2"])
    }
    if (result.status !== 0) {
        rmSync(partial, {force: true})
        fail(`curl exited with ${result.status} fetching ${url}`)
    }
    renameSync(partial, destination)
}

const ensureFile = async (url, destination, expectedBytes, expectedSha) => {
    const existing = statSyncSafe(destination)
    if (!force && existing !== null && existing.size === expectedBytes) {
        console.log(`  [skip] ${destination.replace(REPO_ROOT, ".")} (${existing.size} bytes)`)
        return false
    }
    download(url, destination)
    const actual = statSyncSafe(destination)
    if (actual === null || actual.size !== expectedBytes) {
        fail(`${destination}: expected ${expectedBytes} bytes, got ${actual === null ? "nothing" : actual.size}`)
    }
    if (expectedSha !== undefined) {
        const digest = await sha256Of(destination)
        if (digest !== expectedSha) {fail(`${destination}: SHA-256 mismatch\n  expected ${expectedSha}\n  actual   ${digest}`)}
    }
    return true
}

const extractDf3 = async () => {
    const staging = mkdtempSync(join(tmpdir(), "df3-"))
    const archive = "DeepFilterNet3_onnx.tar.gz"
    download(DF3_TARBALL, join(staging, archive))
    // Relative name with cwd: tar reads "C:\..." as a remote host:path and fails.
    const extract = spawnSync("tar", ["-xzf", archive], {cwd: staging, stdio: "inherit"})
    if (extract.status !== 0) {fail(`tar exited with ${extract.status} extracting ${archive}`)}
    for (const {name} of DF3_FILES) {
        const found = findFile(staging, name)
        if (found === null) {fail(`${name} not found inside the tarball`)}
        renameSync(found, join(DF3_DIR, name))
    }
    rmSync(staging, {recursive: true, force: true})
}

const findFile = (root, name) => {
    for (const entry of readdirSync(root, {withFileTypes: true})) {
        const path = join(root, entry.name)
        if (entry.isDirectory()) {
            const nested = findFile(path, name)
            if (nested !== null) {return nested}
        } else if (entry.name === name) {return path}
    }
    return null
}

const main = async () => {
    console.log("\nAI audio models -> packages/app/studio/public/models/\n")

    if (spawnSync("curl", ["--version"], {stdio: "pipe"}).status !== 0) {fail("`curl` is required but was not found on PATH.")}
    if (spawnSync("tar", ["--version"], {stdio: "pipe"}).status !== 0) {fail("`tar` is required but was not found on PATH.")}
    const python = resolvePython()
    if (python === null) {
        fail("Python 3 with `onnx` and `numpy` is required to derive the DeepFilterNet3 WebNN variants.\n"
            + "        Install them first:  pip install onnx numpy\n"
            + "        (Stem separation does not need Python; only noise suppression on NPU/GPU does.)")
    }
    console.log(`Using Python: ${python}\n`)

    mkdirSync(HTDEMUCS_DIR, {recursive: true})
    mkdirSync(DF3_DIR, {recursive: true})

    console.log("HTDemucs v4 (stem separation)")
    for (const {name, bytes} of HTDEMUCS_FILES) {
        await ensureFile(`${HTDEMUCS_BASE}/${name}`, join(HTDEMUCS_DIR, name), bytes)
    }

    console.log("\nDeepFilterNet3 (noise suppression)")
    const df3Present = DF3_FILES.every(({name, bytes}) => {
        const stats = statSyncSafe(join(DF3_DIR, name))
        return stats !== null && stats.size === bytes
    })
    if (force || !df3Present) {
        await extractDf3()
        for (const {name, bytes, sha256} of DF3_FILES) {
            const path = join(DF3_DIR, name)
            const stats = statSyncSafe(path)
            if (stats === null || stats.size !== bytes) {fail(`${name}: expected ${bytes} bytes`)}
            const digest = await sha256Of(path)
            if (digest !== sha256) {fail(`${name}: SHA-256 mismatch\n  expected ${sha256}\n  actual   ${digest}`)}
            console.log(`  [ok] ${name} (${bytes} bytes, sha256 verified)`)
        }
    } else {
        DF3_FILES.forEach(({name}) => console.log(`  [skip] ${name}`))
    }

    const webnnPresent = DF3_FILES.every(({name}) =>
        statSyncSafe(join(DF3_DIR, name.replace(".onnx", ".webnn.onnx"))) !== null)
    if (force || !webnnPresent) {
        console.log("\nDeriving WebNN variants (GRU -> Scan rewrite)")
        const inputs = DF3_FILES.map(({name}) => join(DF3_DIR, name))
        const rewrite = spawnSync(python, [join(REPO_ROOT, "scripts", "rewrite_gru_for_webnn.py"), ...inputs],
            {stdio: "inherit"})
        if (rewrite.status !== 0) {fail("rewrite_gru_for_webnn.py failed")}
    } else {
        console.log("\n  [skip] WebNN variants already present")
    }

    if (runVerify) {
        console.log("\nVerifying rewrites against originals")
        const verify = spawnSync(python, [join(REPO_ROOT, "scripts", "verify_webnn_rewrite.py")], {stdio: "inherit"})
        if (verify.status !== 0) {fail("verify_webnn_rewrite.py reported failures")}
    }

    console.log("\nDone. Files in packages/app/studio/public/models/:")
    for (const [label, dir] of [["htdemucs_fwd", HTDEMUCS_DIR], ["df3", DF3_DIR]]) {
        for (const entry of readdirSync(dir).sort()) {
            console.log(`  ${label}/${entry}  ${statSync(join(dir, entry)).size} bytes`)
        }
    }
    console.log()
}

await main()
