# AI Audio Features: Stem Separation & Noise Suppression

Two client-side AI audio processing features, both running entirely in-browser via ONNX Runtime
Web with WebNN (NPU/GPU) hardware acceleration, cascading to WASM (CPU) fallback:

- **Stem Separation** — HTDemucs v4, splits a stereo mix into drums/bass/vocals/other.
- **Noise Suppression** — DeepFilterNet3, removes background noise from a single channel.

Related docs: [plans/noise-suppression.md](../plans/noise-suppression.md) (original design doc),
[plans/sdl-release-description.md](../plans/sdl-release-description.md) (release collateral),
[plans/cleanup-todo.md](../plans/cleanup-todo.md) (known follow-ups).

## Shared infrastructure

`packages/app/studio/src/service/ort-shared/`:
- `OrtRuntime.ts` — `loadOrt()` imports `onnxruntime-web/all` (not the default entry — the default
  jsep wasm bundle has an OOB-memory bug when `Scan` nodes sit adjacent to WebNN-partitioned nodes).
  Also `probeBackends()`, `pickPreferredBackend()` (npu > gpu > wasm), `describeBackend()`.
- `OpfsCache.ts` — `readFromOpfs`/`writeToOpfs`/`removeFromOpfs`, model cache via
  `navigator.storage.getDirectory()` (OPFS).
- `FetchWithProgress.ts` — streaming fetch with byte-progress callback and `AbortSignal` support.

`stem-separator/{OrtRuntime,OpfsCache,FetchWithProgress}.ts` are dead one-line re-export shims to
these files, left over from the `ort-shared` extraction — see `plans/cleanup-todo.md`.

## Stem Separation (HTDemucs v4)

| Stage | File / function |
|---|---|
| Entry | `StudioMenu.ts` / `RegionContextMenu.ts` → `showStemSeparatorDialog` in `StemSeparatorDialog.tsx` |
| Orchestrator | `StemSeparator.ensureModelLoaded` / `.separate` / `.separateFrames` |
| Model fetch | `ModelLoader.loadHtdemucsFwdModel` (graph ~2.4MB → weights ~176MB) |
| Session create | inline in `StemSeparator` — single session, no race condition to guard against |
| Decode/resample | `WavDecoder.decodeFileToStereo44100` / `resampleFramesToStereo44100` (`OfflineAudioContext`) |
| Framing | `Chunker.chunkStereo` — `SEGMENT_LENGTH=343980`, `OVERLAP=171990` (Constants.ts) |
| Transform | `Stft.computeStftChannel`, radix-2 FFT (`Fft.ts`) |
| Features | `PreForward.htdemucsPreForward` — mean/std norm, builds both `x` (freq/CAC) and `xt` (time) tensors |
| Inference | one call: `session.run({x, xt})` — outputs 4 stems × 2 channels for both branches at once |
| Inverse transform | `PostForward.htdemucsPostForward` — de-norm, `computeIstftChannel` per stem/channel, then **adds** the time-domain branch output directly (hybrid dual-branch fusion) |
| Reassembly | triangular-weighted overlap-add across segments (`buildTriangularWeights`), final divide by weight sum |
| Write-back | `StudioService.importSeparatedStems` → `#importSampleAsTapeTrack` ×4 — always creates 4 new Tape tracks, never touches the source |

## Noise Suppression (DeepFilterNet3)

| Stage | File / function |
|---|---|
| Entry | `StudioMenu.ts` / `RegionContextMenu.ts` (passes `sourceRegion`) → `showNoiseSuppressorDialog` in `NoiseSuppressorDialog.tsx` |
| Orchestrator | `NoiseSuppressor.ensureModelLoaded` / `.processFrames` |
| Model fetch | `ModelLoader.loadDeepFilter3Models` — 3 ONNX graphs (`enc`, `erb_dec`, `df_dec`) fetched in **parallel** |
| Model variant | `variantForBackend` picks "original" vs `.webnn`-suffixed graphs (rewritten by `scripts/rewrite_gru_for_webnn.py` for GRU-op compatibility) — these are genuinely different ONNX files, not the same graph on a different EP |
| Session create | `NoiseSuppressor.#getOrCreateSessions` — 3 sessions, created **sequentially** (see Known Issues) |
| Resample | `Resampler.resampleTo48k` (`OfflineAudioContext`) — each channel processed independently |
| Framing | `FRAME_SIZE=480` (10ms @ 48kHz) |
| Transform | `Analysis.frameAnalysis` — Vorbis window + `BluesteinFft` (needed since `FFT_SIZE=960` isn't a power of 2) |
| Features | `ErbFilterbank.computeErb` + `Normalizer.normalizeErb` (dB-domain EMA, `ATTEN_DENOM=40.0`) / `normalizeSpec` (magnitude EMA) |
| Inference | 3 calls in sequence: `encoder.run` → produces `emb` → `erbDecoder.run({emb,...})` (32-band gain mask) + `dfDecoder.run({emb,...})` (5-tap complex DF coefficients for first `NB_DF=96` bins) |
| Mask/filter | `ErbFilterbank.expandMask` (step-function, **not interpolated**, bins ≥96) + `DfFilter.applyDfFilter` (5-tap complex FIR via ring buffer, bins <96) |
| Post-filter | unconditional post-filter stage (`beta=0.02`) after masking — see Known Issues |
| Inverse transform | `Synthesis.frameSynthesis` — save-and-add, reuses the **same** `BluesteinFft` instance from analysis, clamps to `[-1,1]` |
| Write-back | with `sourceRegion`: `StudioService.importDenoisedRegion` (**deletes the source region**, see Known Issues); without: `importDenoisedAsTapeTrack` → `#importSampleAsTapeTrack`, new Tape track |

## Model provenance

- **HTDemucs v4** — exported from Meta Research's pretrained `htdemucs.th` checkpoint (MIT license)
  via `Music-Source-Separation-Training/openvino_conversion/convert_htdemucs_fwd_only.py` — see
  [HTDemucs v4 conversion steps](#htdemucs-v4-conversion-steps) below.
- **DeepFilterNet3** — `enc.onnx` / `erb_dec.onnx` / `df_dec.onnx` / `config.ini` checked in at
  `deepfilternet2-demo/models/df3/` are byte-for-byte identical (SHA-256 match) to
  `DeepFilterNet3_onnx.tar.gz`, an ONNX export dated 2023-05-23 from the upstream
  [Rikorose/DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) project (MIT license).
  `config.ini`'s `[df]`/`[deepfilternet]` sections (`sr=48000`, `fft_size=960`, `hop_size=480`,
  `nb_erb=32`, `nb_df=96`, `df_order=5`, `df_lookahead=2`) match
  `noise-suppressor/pipeline/Constants.ts` exactly, confirming this is the model in production use.
  Notably `config.ini` has `mask_pf = False` — the checkpoint itself was trained/exported with its
  post-filter **disabled**; the post-filter applied in `NoiseSuppressor.ts` (`beta=0.02`) is
  entirely openDAW's own application-layer addition, not part of the exported model.

### HTDemucs v4 conversion steps

No training or fine-tuning was involved — this is a format translation of an already-trained
checkpoint, so no dataset was used. (`config_musdb18_htdemucs.yaml` names MUSDB18 only because
that's what Meta trained on; the script reads it for architecture hyperparameters.)

**Inputs**
- Checkpoint `955717e8-8726e21a.th` — Meta's pretrained base `htdemucs.th` (non-fine-tuned).
- Config `configs/config_musdb18_htdemucs.yaml`.
- Model class `models/demucs4ht.py` from the `RyanMetcalfeInt8/Music-Source-Separation-Training`
  fork (`openvino_conversion` branch) — exposes the split `pre_forward`/`fwd`/`post_forward`
  methods that make a partial export possible.

**Environment** — `pip install -r requirements.txt` plus `onnx openvino onnxscript`. Key packages:
`demucs==4.0.0`, `torch`/`torchaudio`, `omegaconf`, `antlr4-python3-runtime==4.9.3`, `einops`,
`rotary_embedding_torch`, `ml_collections`, `openunmix`, `onnxruntime`.

**Command** (per [htdemucs-demo/HANDOFF.md](../htdemucs-demo/HANDOFF.md)):
```
PYTHONIOENCODING=utf-8 python convert_htdemucs_fwd_only.py \
  --config .../configs/config_musdb18_htdemucs.yaml \
  --checkpoint .../955717e8-8726e21a.th \
  --output .../htdemucs-demo/models/htdemucs_fwd/htdemucs_fwd.onnx
```

**What `convert_htdemucs_fwd_only.py` does**

| Step | Detail |
|---|---|
| 1. Patch for tracing | Monkeypatches `pad1d` in `demucs.hdemucs` and `models.demucs4ht` to drop an assertion that trips dynamo's data-dependent shape check |
| 2. Build model | Reads the YAML, coerces scientific-notation strings to floats, constructs `HTDemucs(...)` |
| 3. Load weights | `torch.load` the `.th`, unwrap the `'state'` key, `load_state_dict(strict=False)` |
| 4. Get example tensors | Runs the model's own `pre_forward` on a zeros tensor `[1, 2, 343980]` to obtain correctly-shaped `(x, xt)` |
| 5. Wrap `fwd` only | `FwdWrapper.forward(x, xt) → model.fwd(x, xt)` — deliberately excludes STFT/iSTFT |
| 6. Dry-run check | Verifies the wrapped forward pass works in Python before export |
| 7. Export | `torch.onnx.export(..., input_names=['x','xt'], output_names=['x_out','xt_out'], dynamo=True, opset=17)` |

Output: `htdemucs_fwd.onnx` (2.3MB graph) + `htdemucs_fwd.onnx.data` (168MB external weights).

**Why fwd-only, not the full pipeline** — `torch.onnx.export` cannot trace the complex-tensor ops
(`view_as_complex`, complex `Pad`) used by HTDemucs's `_mask`/`_ispec`. STFT/normalize and
iSTFT/de-normalize/branch-sum/trim were therefore reimplemented in TypeScript (`PreForward.ts` /
`PostForward.ts`), matching Intel's own C++ approach in `mod-openvino/htdemucs.cpp`. The sibling
script `convert_htdemucs_to_onnx.py` attempts the full pre+fwd+post export and hits exactly this
wall — it is not the path used.

**Optional post-step** — `decompose_gelu.py in.onnx out.onnx` rewrites native `Gelu` as
`x * 0.5 * (1 + Erf(x/√2))` for WebNN compatibility. Only needed if WebNN-NPU fails to load with
"dataTypes undefined"; not required on Chrome 148+ with DirectML.

**Validation** — `htdemucs-demo/reference/gen_reference.py` pushes a sine-wave input through
`pre_forward → ONNX fwd → post_forward` in Python, dumping every intermediate tensor; the demo UI
diffs each JS stage against these. Recorded: STFT maxDiff 9.5e-7, pre_forward 7.6e-6, ONNX fwd MAE
9e-4 (fp16), post_forward 1.8e-7.

## Known issues / fragile code

- **WebNN session-creation race condition.** `NoiseSuppressor.#getOrCreateSessions` creates its 3
  sessions sequentially with a comment documenting an ORT-Web + WebNN/OpenVINO EP crash
  ("memory access out of bounds") when `Scan` nodes sit next to WebNN-partitioned nodes, if created
  in parallel. Do not "optimize" this back to `Promise.all` — the model *fetch* stage in
  `ModelLoader.ts` is unaffected and stays parallel; only session *creation* must stay sequential.
- **Bluestein FFT inverse direction.** `BluesteinFft.inverse()` computes
  `IFFT(X) = conj(FFT(conj(X))) / n` specifically to avoid inverse-direction chirp/kernel
  conjugation bookkeeping that a prior implementation got wrong (per its own comment). Don't
  simplify this back toward a direct inverse-chirp implementation.
- **"Non-destructive" claim contradicted by code.** Both `plans/noise-suppression.md` ("Original
  region is untouched") and `plans/sdl-release-description.md` ("Processing is non-destructive —
  new regions are created while originals are preserved") state the original region must survive.
  `StudioService.importDenoisedRegion` actually calls `sourceRegion.box.delete()` before creating
  the denoised region at the same position, inside one `editing.modify()` transaction (undo-able,
  but not non-destructive). Needs a decision: fix the code, or update the docs.
- **Noise suppressor output is stereo at the source sample rate, not "single-channel at 48kHz."**
  `plans/sdl-release-description.md`'s architecture diagram labels the noise-suppressor output as
  "Single-channel output at 48 kHz" — both parts are wrong. `NoiseSuppressor.processFrames` handles
  `numChannels = Math.min(frames48k.length, 2)` (up to stereo, each channel run through the model
  independently; a mono source just gets `right` duplicated from `left`), and the final
  `resampleFrom48k(outputChannels, sampleRate)` converts back to the **source's original sample
  rate** before returning. 48kHz is only DeepFilterNet3's internal processing rate.
- **Post-filter shipped despite being a stated non-goal.** `plans/noise-suppression.md`'s
  "Non-goals for this pass" says the post-filter should ship off with no UI control. It's shipped
  **on**, unconditionally, with no toggle (see Model provenance — the checkpoint itself ships with
  `mask_pf = False`, so this is purely an openDAW-side addition, not model behavior).
- **Reconstruction schemes are not interchangeable.** Stem separation uses classic overlap-add
  with triangular crossfade weights + final `winSum` division. Noise suppression uses save-and-add
  with complementary Vorbis analysis/synthesis windows and ring-buffer carryover (no `winSum`
  needed). Don't port logic between the two files.
- **Backend/model-variant coupling unclear.** Switching WebNN ↔ WASM for noise suppression means
  loading different ONNX files (`variantForBackend`), not just re-pointing one session at a different
  EP. It's unclear from the code whether cached sessions are invalidated on a runtime backend
  change — a stale session against the wrong model variant would misbehave silently rather than
  error. Worth confirming directly with whoever wrote `#getOrCreateSessions`.
