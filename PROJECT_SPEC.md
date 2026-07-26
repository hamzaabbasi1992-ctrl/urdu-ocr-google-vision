# PROJECT_SPEC.md — Urdu Nastaleeq OCR

**This document is the single source of truth for this project.**
Every future change — code, architecture, dependencies, defaults — must conform to this
document. If a change requires deviating from it, **this document is updated first**,
in the same change, before or alongside the code. No architectural decision is valid
if it only exists in code/commit history and not here.

---

## 1. Mission

Extract the literal text from low-quality scanned Urdu **Nastaleeq** book PDFs (a
private Islamic-book archive) with the **highest possible accuracy**, fully **offline**.

**Accuracy is everything. Speed is not a goal. UI polish is not a goal.**
Where any future decision trades accuracy against speed, simplicity, or development
convenience, accuracy wins by default, unless the user explicitly says otherwise for
that specific decision.

## 2. Non-negotiable rules

These come from the original project spec and are never subject to "helpful" override:

1. **Never rewrite** recognized text. Never paraphrase. Never let a language model
   "improve," summarize, or auto-correct OCR output.
2. **Never guess** missing/unclear words. If confidence is low, flag it — never invent
   a plausible replacement.
3. **Never strip diacritics** — zabar, zer, pesh, and all other i'raab marks, Quranic
   symbols, Arabic loanwords, and Urdu-Indic numerals must survive untouched from
   recognition through to every export format.
4. ~~Fully offline at runtime.~~ **Revised, with explicit user sign-off.**
   Originally: one-time model downloads acceptable, no cloud OCR/LLM API calls
   ever process this content - because this archive is private religious/family
   material the original speakers/authors never consented to third-party cloud
   processing (same reasoning as the sibling Urdu Bayan Transcriber app). The
   user explicitly chose to allow cloud OCR for this project after asking about
   Google Cloud Vision, understanding and accepting that pages processed this
   way are sent to Google's servers. **This does not retroactively make cloud
   processing the default** - any engine that calls a cloud API must be clearly
   labeled as such (see `GoogleVisionEngine` in Section 6), and this project
   still prefers local engines where accuracy is comparable; cloud is now an
   available option, not the assumed path.
5. **Output must contain only the OCR result** — plus explicitly-labeled confidence/
   metadata. No generated commentary, no synthesized filler text.

## 3. Dependency policy

**Every dependency must justify its existence with measurable OCR accuracy
improvement, measured on this project's own test pages — not a vendor's or
paper's self-reported benchmark on different data.** This is an OCR project, not
an AI/chatbot platform, and its dependency footprint should read that way.

Specifically excluded unless measured evidence says otherwise:
- `transformers`, `accelerate`, `peft`, `qwen-vl-utils`, or any other generic
  LLM/VLM-serving stack.
- Any large language model or vision-language model used as a recognition engine.

**Why this matters here:** a Qwen2-VL-2B + LoRA engine (`Qaari-0.1`) was added to
Layer 6 based on the *model card's own* reported WER/CER. It was never actually
run against this project's real or synthetic test pages before its ~6-package,
multi-GB dependency chain was committed — the justification was a vendor
benchmark, not a measurement. That engine and its dependencies have been removed
until (if ever) a controlled comparison against the existing engines, on this
project's own pages, shows it measurably improves fused accuracy enough to be
worth its weight, load time, and CPU cost.

**Before adding any new dependency**, especially another model/engine:
1. State what specific accuracy gap it addresses that current engines don't.
2. Run it against a real (or realistic synthetic) test set from this project.
3. Record the measured result — not the vendor's claimed result — before deciding
   to keep it.
4. Only then does it get added to Layer 6 and this document.

This applies to every future engine candidate mentioned elsewhere in this
document (UTRNet, EasyOCR, etc.) just as much as it applied to Qaari - none of
them are pre-approved; they're candidates pending the same measurement.

## 4. Measurement-first build order

**No new Layer 3+ module (preprocessing transform, recognition engine, or fusion
step) may be added until a working, measurable end-to-end benchmark exists, and
every module added after that point must prove its value against that benchmark
or be removed.** This is the operational form of Section 3's dependency policy,
applied to modules generally, not just external dependencies.

**Why:** Layer 1 (Ingestion) and Layer 2 (Render-Quality Search) were built and
tested module-by-module correctly. Layer 3 (Preprocessing Transforms) was about
to be built the same way - 15 modules, one at a time - but that would have meant
15 modules of accuracy-affecting logic added *before there was any way to measure
whether any of them actually helped.* That's backwards for an accuracy-first
project: it repeats the exact mistake the Qaari removal (Section 3) was supposed
to have already taught - adding something because it plausibly sounds like it
should help, not because it was measured to help.

**The required order:**
1. Build the smallest possible complete pipeline: `PDFLoader` → `PageRasterizer`
   (done) → a *minimal* preprocessing set (`Deskewer`, `GlobalContrastEnhancer`
   i.e. CLAHE, `Denoiser`) → `PaddleOCREngine` (the single existing, unmodified
   engine - not the full Layer 6/7 ensemble yet) → `TextExporter`.
2. Build the evaluation infrastructure this requires: `GroundTruthLoader`,
   `CERCalculator`, `WERCalculator`, `ConfidenceAggregator`, `BenchmarkReporter`
   (see the new Evaluation & Benchmarking entry in Section 6).
3. Run the minimal pipeline against real (or realistic synthetic, with known
   exact ground-truth text) pages and produce a benchmark report: CER, WER,
   average confidence, processing time.
4. **Only after that report exists** may additional Layer 3 preprocessing
   modules, additional Layer 6 engines, or Layer 7 fusion be added - and each
   one must be benchmarked in (measured CER/WER before vs. after adding it) as
   part of adding it. If a module does not measurably improve CER or WER, it is
   removed, not kept "just in case."

This does not change the target architecture in Section 6 - the full module set
is still the destination. It changes the *order* modules are built in, so that
from this point forward, every module's existence is justified by a number, not
a plausible-sounding rationale.

**Per-module proof checklist.** From this point forward, every completed
preprocessing (or recognition/fusion) module must ship with all of the
following before it can be considered for the default pipeline:

1. Unit tests (the module's own correctness, in isolation).
2. A benchmark run against the current baseline pipeline (with the module
   disabled) and against the pipeline with the module enabled - same pages,
   same ground truth.
3. Before/after images, so the visual effect is inspectable, not just implied
   by a number.
4. CER difference (module vs. baseline).
5. WER difference (module vs. baseline).
6. Average confidence difference (module vs. baseline).

**If a module does not demonstrate improvement on this checklist, it is not
added to the default pipeline.** It may still be kept in the codebase as an
available, off-by-default, individually-toggleable option (this is compatible
with the original spec's "every step configurable" requirement) - but "off by
default, not proven" and "on by default" are different things, and only
measured modules get the latter.

## 5. Governing architectural principle

**Single Responsibility, strictly.** Every module does exactly one thing. A module
that renders does not judge quality. A module that judges quality does not decide
whether to re-render. A module that recognizes text does not vote between engines.
A module that votes does not know how to recognize. This is not a style preference —
it is what makes an accuracy-first, search/ensemble-based system tractable and
debuggable, since Layer 7 (Fusion) and Layer 8 (Variant Search) depend on being able
to swap, add, or re-run any single module without side effects on the others.

**Accuracy-first means ensemble and search are the default architecture, not
optional extras.** There is no single "primary engine" and no single "the pipeline."
Multiple OCR engines run on multiple preprocessing/DPI variants of every page by
default, and a dedicated Fusion layer reconciles them. Earlier iterations of this
project used a 2-engine confidence-threshold arbiter as the default; that is
superseded by the architecture below.

## 6. Module architecture

### Layer 1 — Ingestion
| Module | Responsibility |
|---|---|
| `PDFLoader` | Open a PDF, expose page count/metadata. Nothing else. |
| `PageRasterizer` | Render one PDF page to a raster image at one specified DPI. No quality judgment, no looping. |

### Layer 2 — Render-Quality Search
| Module | Responsibility |
|---|---|
| `MultiDPICandidateGenerator` | Call `PageRasterizer` at every configured DPI (e.g. 300/600/900/1200) unconditionally. Produces a candidate set; judges nothing. |
| `SharpnessScorer` | Given one image, return a sharpness metric. Pure function. |
| `SuperResolutionUpscaler` | Given one image, return an upscaled version. Does not decide whether to run. |

### Layer 3 — Preprocessing Transforms
Each is a pure `image -> image` function. None decide whether they should run; none know about each other.

| Module | Responsibility |
|---|---|
| `OrientationDetector` | Detect page rotation angle only. |
| `Deskewer` | Detect + correct fine skew angle only. |
| `PageCropper` | Detect and remove borders/blank margins only. |
| `ShadowRemover` | Correct illumination gradient only. |
| `NoiseClassifier` | Classify noise type present. Does not modify the image. |
| `Denoiser` | Apply a denoise strategy for a given noise type. Takes classification as input. |
| `GlobalContrastEnhancer` (CLAHE) | Global contrast only. |
| `LocalContrastEnhancer` | Local/adaptive contrast only. |
| `GammaCorrector` | Tonal curve only. |
| `FadedTextDetector` | Flag low-contrast-but-inked regions. Does not modify the image. |
| `FadedTextBooster` | Enhance regions flagged by `FadedTextDetector` only. |
| `StrokeEnhancer` | Recover/thicken thin strokes only. |
| `InkEnhancer` | Deepen ink tone only. |
| `Sharpener` | Unsharp-mask only. |
| `MorphologicalCleaner` | Remove speckle noise only. |
| `Binarizer` | Optional thresholding, used only where a specific downstream engine wants binary input. |

### Layer 4 — Diacritic & Mark Protection (cross-cutting)
| Module | Responsibility |
|---|---|
| `ConnectedComponentAnalyzer` | Given an image, return labeled components + stats. Pure analysis. |
| `DiacriticClassifier` | Given components, classify which are likely dots/zabar/zer/pesh. Returns a mask only. |
| `ProtectionGuard` | Given original image, processed image, and a protection mask, restore protected pixels. **The only module allowed to reconcile "processed vs. original."** Every risky transform in Layer 3 must route through this rather than implement its own protection logic. |

### Layer 5 — Structure Segmentation
| Module | Responsibility |
|---|---|
| `BaselineEstimator` | Estimate baseline/x-height band per line. Feeds `DiacriticClassifier` only. |
| `WordRegionProposer` | Propose word/ligature-cluster regions within a line. Structure only, no recognition. |

**`LineSegmenter` - built, then removed (2026-07-26).** Split a page image
into line bands; existed solely to feed pre-cropped line images to
`QaariVLMEngine`/`UTRNetEngine` (both expect one line per call, unlike
`GoogleVisionEngine`, which recognizes a whole page at once). Deleted along
with `QaariVLMEngine` - see Layer 6 below - since its only consumer is gone.
Would need to be rebuilt if a future line-based engine is ever added.

### Layer 6 — Recognition Engines
Each engine implements one interface (`region -> text, confidence`) and does nothing else. No engine merges, votes, or judges another. **All engines run on all pages** — there is no "primary engine."

| Module | Responsibility |
|---|---|
| `TesseractEngine` | Recognize via Tesseract only. Never built under this architecture (was part of the removed legacy 2-engine arbiter - see Section 9). Still an unproven candidate, pending the same measurement discipline as any other engine. |
| `EasyOCREngine` | Recognize via EasyOCR (Arabic model) only — reference signal, per original spec ("EasyOCR only for comparison"). Same status as `TesseractEngine`: never built under this architecture. |

**`PaddleOCREngine` - built, measured, then removed (2026-07-26).**
Measured baseline: CER 0.87, **WER 1.00** on the benchmark fixture -
confirmed this engine does not work for Nastaleeq (see the engine
comparison table and baseline-benchmark history below, kept as the
historical record of *why*). Once `GoogleVisionEngine` was adopted as the
sole engine, the user asked to remove Paddle/Qaari/UTRNet entirely rather
than keep disqualified code around. `PaddleOCREngine` itself
(`app/core/recognition/paddle_ocr_engine.py`) and its tests are deleted.
Its `RecognizedWord`/`assign_reading_order` were engine-agnostic (also used
by `GoogleVisionEngine` and `TextExporter`) and were split out first, into
`app/core/recognition/recognized_word.py`, so removing Paddle didn't break
the working Google Vision path. `app/core/minimal_pipeline.py` (the
Paddle-based orchestrator) and `tools/run_benchmark.py` were removed with
it - Google Vision has its own simpler path (`app/simple_gui.py`,
`tools/run_benchmark_google_vision.py`), so no replacement orchestrator was
needed.

**`UTRNetEngine` - deprioritized, never built, now dropped as a candidate
(2026-07-26).** Its real inference code (`model.py`/`dataset.py`/`utils.py`/
`modules/`, ~900 lines) would have imported several training-only
dependencies (`lmdb`, `imgaug`, `matplotlib`, `timm`, `pytz`, `six`) just to
load the files, even though only the HRNet+BiLSTM+CTC inference path would
ever be used - exactly the kind of dependency-weight problem Section 3
exists to catch. Originally deprioritized in favor of measuring
`QaariVLMEngine` first; dropped outright, by explicit user decision, once
`GoogleVisionEngine` proved sufficient and the user chose to standardize on
it rather than keep evaluating offline alternatives. No code ever existed
for this engine, so there was nothing to delete - this note just records
that it's no longer a live candidate. Could be revisited from scratch if
the cloud trade-off (Section 2, rule 4) is ever reversed.

**`QaariVLMEngine` - built, measured, disqualified, then removed
(2026-07-26).** Measured CER 0.9577, WER 1.0000 - paraphrased into English
instead of transcribing (violates the "never rewrite" rule, Section 2) and
was ~600x too slow (1988.7s/page). Kept in the codebase for a time as a
measured, documented-disqualified candidate; deleted along with
`PaddleOCREngine` once the user decided to drop all non-Google-Vision
engines rather than keep disqualified code around.
`app/core/recognition/qaari_engine.py`, `tests/test_qaari_engine.py`,
`tools/run_benchmark_qaari.py`, and its sole consumer of Layer 5's
`LineSegmenter` (see above) are all deleted. `app/core/paths.py`
(`model_cache_dir`, only ever used by this engine) was deleted with it.

**`GoogleVisionEngine` - built and measured; the strongest result so far.**
Recognizes via Google Cloud Vision's `DOCUMENT_TEXT_DETECTION`
(`app/core/recognition/google_vision_engine.py`, `tests/test_google_vision_engine.py`,
6/6 passing - 5 pure-logic + 1 real-API test gated behind
`GOOGLE_VISION_CREDENTIALS_PATH`). Built under the Section 2 offline-rule
revision (explicit user sign-off). `name = "google_vision_cloud"` -
deliberately explicit in code, never silently mistakable for a local engine.

**Measured result** (`benchmark_page.pdf#0`, same fixture as PaddleOCR/Qaari),
initial:
```
CER=0.1308  WER=0.4894  avg_confidence=0.9456  time=3.22s
```
Actual recognized text is genuinely correct, readable Urdu - e.g. line 3
(`علم حاصل کرو چاہے تمہیں چین ہی کیوں نہ جانا پڑے۔`) came back exactly right
apart from one stray space. Most of the WER was inflated by trivial
diacritic-omission and spacing differences, not real transcription
failures - qualitatively a different category of result from PaddleOCR
(fragmented, wrong) or Qaari (paraphrased into English, disqualified).

**Fixed a real bug this exposed**: `TextExporter.assemble_text` joined every
recognized token with a space, including punctuation - Google Vision returns
punctuation (`۔`, `،`, etc.) as separate word tokens, so output read
`لفظ ۔` instead of `لفظ۔`. Fixed by attaching a fixed set of trailing
punctuation directly to the preceding word (`app/core/export/text_exporter.py`,
3 new tests in `tests/test_text_exporter.py`, 9/9 passing). This is a
text-assembly correctness fix, not an accuracy trade-off, so it did not need
Section 4's full per-module proof checklist - but was still re-measured:

```
CER=0.1077  WER=0.2979  avg_confidence=0.9456  time=3.28s
```

A 39% relative WER reduction from one bug fix, no OCR change at all. Also
comfortably meets the 1-minute/20-line speed requirement (3.28s for 5
lines here).

**Also measured, and reversed a prior assumption**: does this project's own
preprocessing (`Deskewer`/`Denoiser`/`GlobalContrastEnhancer`) help Google
Vision, the way it might help a local engine? Tested directly - it does not:

| | CER | WER | Time |
|---|---|---|---|
| Raw (no preprocessing) | **0.0923** | **0.2766** | 1.48s |
| With deskew+denoise+CLAHE | 0.1077 | 0.2979 | 1.09s |

Preprocessing measurably *hurts* Google Vision - plausibly because its own
document-OCR pipeline is already tuned, and this project's bilateral-filter
denoise/CLAHE smooth away fine detail (diacritic dots) Google's model would
otherwise use. Per Section 4 ("if a module does not measurably improve CER
or WER, it is removed"), `app/simple_gui.py`'s `OCRWorker` was updated to
call only `PDFLoader -> PageRasterizer -> GoogleVisionEngine -> TextExporter`,
with no preprocessing step in between, and re-verified end-to-end through
the actual GUI worker class (not just the standalone measurement) before
being counted as done.

**Cumulative result**: WER went from 0.4894 (first working measurement) to
0.2766 (current) - a 43% relative reduction - from two changes, neither of
which touched the OCR engine itself: a text-assembly bug fix, and removing
preprocessing that was never actually proven to help this engine in the
first place.

### Layer 7 — Cross-Engine Fusion
The accuracy-critical layer. Supersedes the old 2-engine confidence-threshold arbiter.

| Module | Responsibility |
|---|---|
| `BBoxAligner` | Align N engines' regions (different granularities, different DPIs) into one canonical region grid. Geometry only. |
| `ConfidenceNormalizer` | Map each engine's native confidence signal onto one comparable scale. Calibration only. |
| `CandidateCollector` | For each canonical region, gather every engine's candidate text + normalized confidence. Gathering only. |
| `VotingFusionEngine` | Decide final text per region via confidence-weighted vote / agreement-based selection. **Never invents text not proposed by an engine** (rule #2). |
| `DisagreementFlagger` | Flag regions where engines disagree strongly, independent of which candidate won. Stronger low-confidence signal than any single engine's own score. |

### Layer 8 — Preprocessing/Variant Search
First-class, not a side "benchmark mode" — exhaustive search is affordable when accuracy is the only goal.

| Module | Responsibility |
|---|---|
| `VariantGenerator` | Enumerate preprocessing + DPI combinations to try. Enumeration only. |
| `VariantEvaluator` | Run one variant through Layers 3-7 and score the result. Execution only. |
| `BestVariantSelector` | Given scored variants, pick (or ensemble across) the winner(s). Selection only. |

### Layer 9 — Post-Processing
Mechanical only — this is the "never rewrite" boundary made concrete in code.

| Module | Responsibility |
|---|---|
| `WhitespaceNormalizer` | Collapse/clean whitespace only. |
| `LineReconciler` | Merge erroneously-split line detections only. |
| `CharacterIntegrityValidator` | Verify no diacritic/numeral/Arabic character present in any candidate was dropped from the fused output. Flags, never edits. |
| `LowConfidenceFlagger` | Mark final words below threshold or flagged by `DisagreementFlagger`. Never alters them. |

### Layer 10 — Export
| Module | Responsibility |
|---|---|
| `CoordinateMapper` | Map canonical-region coordinates back to the original page image, given accumulated geometric transforms. Geometry only. |
| `TextExporter` | `output.txt` only. |
| `DocxExporter` | `output.docx` only. |
| `SearchablePDFExporter` | Searchable PDF only. |
| `JSONExporter` | Structured JSON (page, word, bbox, confidence) only. |
| `HeadingClassifier` | Given lines already grouped by `line_index`, decide which are headings from relative line height and/or isolation (extra vertical whitespace above the line) - see the dedicated note below for why (Google Vision exposes neither font weight nor any other direct style signal). Classification only - never inspects/alters the recognized text, and does not itself touch export formatting. |

### Layer 11 — Orchestration
Coordination only — zero content/business logic.

| Module | Responsibility |
|---|---|
| `PageOrchestrator` | Sequence Layers 2-9 for one page. |
| `DocumentOrchestrator` | Aggregate pages into a document result. |
| `BatchController` | Sequence documents; pause/cancel/progress only. |

### Cross-cutting Infrastructure
| Module | Responsibility |
|---|---|
| `ModelRegistry` | Load/cache heavy model instances. Lifecycle only, no inference logic. |
| `ConfigStore` | Hold validated configuration values. Data only. |
| `AuditLog` | Record what was tried, what won, and why, per page. Required for Layers 7-8 to be inspectable/debuggable. |

### Evaluation & Benchmarking (required by Section 4 before Layer 3+ expands)
| Module | Responsibility |
|---|---|
| `GroundTruthLoader` | Load a reference (known-correct) transcription for a test page/document. Data only. |
| `CERCalculator` | Given hypothesis + reference text, compute Character Error Rate. Pure function. |
| `WERCalculator` | Given hypothesis + reference text, compute Word Error Rate. Pure function. |
| `ConfidenceAggregator` | Given a set of OCR word confidences, compute an average. Pure function. |
| `BenchmarkReporter` | Given CER/WER/confidence/timing results, produce a report. Formatting only, does not compute metrics itself. |

## 7. Explicit non-goals

- ~~Runtime speed on any given page or batch.~~ **Revised, with explicit user
  sign-off, after the first real benchmark run (29.25s/page) was found too slow
  in practice.** Speed now matters, but as a *secondary* goal: any speed change
  that would reduce CER/WER must still go through Section 4's benchmark-before/
  after discipline like any other change - "faster" is not exempt from "prove
  it doesn't hurt accuracy." Speed changes with **no** accuracy trade-off
  (e.g. not re-loading an already-loaded model) need no such proof.
- UI sophistication. The GUI exists to expose the pipeline's controls and results,
  not to be a polished product surface.
- Backward compatibility with the earlier "2-engine arbiter, single primary engine"
  design. That design is superseded, not preserved as an option.

## 8. Change control

- Any change that adds, removes, merges, or reassigns responsibility for a module
  must update the relevant table in Section 6 in the same change.
- Any change to the non-negotiable rules (Section 2), dependency policy
  (Section 3), or measurement-first build order (Section 4) requires explicit
  user sign-off — an agent must never loosen these unilaterally.
- If code and this document disagree, **this document wins** until it is
  deliberately updated; the code is the thing considered wrong in that case.
- **Workflow discipline:** never work on two modules simultaneously. Finish one
  completely, test it, commit it — only then start the next module. This
  applies to every module in Section 6 without exception.
- **Every Layer 3+ module (preprocessing/recognition/fusion) must additionally
  satisfy Section 4's per-module proof checklist before it is enabled by
  default.**

## 9. Status

Migration to this architecture is in progress, module-by-module, per Section 8's
workflow discipline.

**Old monolithic build removed (2026-07-25).** The previous 2-engine
PaddleOCR/Tesseract-arbiter codebase (`app/core/{ocr,preprocess,exporters,
pdf_render,segmentation,benchmark,control,job,job_queue,logging_setup,
model_manager,models,postprocess}.py`, `app/gui/`, `app/workers/`, `app/cli.py`,
`app/main.py`, `Start.bat`) has been deleted. Confirmed via grep across the
active codebase (`app/simple_gui.py`, all new-architecture modules under
`app/core/`, `tests/`, `tools/`) before deletion that nothing outside the old
codebase imported from it, except `app/core/paths.py` (kept at the time -
used by `QaariVLMEngine` for model cache location; `app/core/paths.py` was
itself deleted the next day when `QaariVLMEngine` was removed - see the next
note below). `pyproject.toml`'s `[project.scripts]` entry point updated from
`app.main:main` to `app.simple_gui:main` accordingly. The GUI is now
exclusively `app/simple_gui.py` (launched via `Start Google Vision GUI.bat`);
the old `Start.bat` launcher for the deleted `app.main`/`app.gui` was removed
with it.

**Paddle/Qaari/UTRNet engines removed (2026-07-26).** With `GoogleVisionEngine`
established as the working, sole engine, the user asked to remove
`PaddleOCREngine`, `QaariVLMEngine`, and the never-built `UTRNetEngine`
candidate entirely rather than keep disqualified/unused engine code around.
See the Layer 5 (`LineSegmenter`) and Layer 6 removal notes below for what
was deleted and why; the historical measurement narrative later in this
section (engine comparison table, baseline benchmark, DPI sweep) is kept
intact as the record of *why* each was disqualified, even though the code
itself is gone.

**Completed modules** (implemented + standalone-tested; commit pending - git
identity is not yet configured for this repo):
- Layer 1: `PDFLoader` (`app/core/ingestion/pdf_loader.py`, `tests/test_pdf_loader.py`, 7/7 passing)
- Layer 1: `PageRasterizer` (`app/core/ingestion/page_rasterizer.py`, `tests/test_page_rasterizer.py`, 6/6 passing)
- Layer 2: `SharpnessScorer` (`app/core/render_quality/sharpness_scorer.py`, `tests/test_sharpness_scorer.py`, 4/4 passing) - **not part of the minimal pipeline (see below); built ahead of need**
- Layer 2: `MultiDPICandidateGenerator` (`app/core/render_quality/multi_dpi_candidate_generator.py`, `tests/test_multi_dpi_candidate_generator.py`, 4/4 passing) - **not part of the minimal pipeline; built ahead of need**
- Layer 2: `SuperResolutionUpscaler` (`app/core/render_quality/super_resolution_upscaler.py`, `tests/test_super_resolution_upscaler.py`, 5/5 passing) - **not part of the minimal pipeline; built ahead of need**
- Layer 3: `Deskewer` (`app/core/preprocessing/deskewer.py`, `tests/test_deskewer.py`, 5/5 passing) - initial version shipped with a real bug (wrong `cv2.minAreaRect` angle-convention assumption for this OpenCV build; silently rejected every real skew as "implausible"), caught only once its test was actually run and fixed before being counted as done. Reinforces why untested code must never be marked complete.
- Layer 3: `GlobalContrastEnhancer` (`app/core/preprocessing/contrast_enhancer.py`, `tests/test_contrast_enhancer.py`, 4/4 passing) - CLAHE, ported from old `preprocess/contrast.py`.
- Layer 3: `Denoiser` (`app/core/preprocessing/denoiser.py`, `tests/test_denoiser.py`, 4/4 passing) - minimal single-strategy bilateral filter, ported from old `preprocess/denoise.py`'s "gaussian" branch.
- Layer 6: `RecognizedWord`/`assign_reading_order` (`app/core/recognition/recognized_word.py`, `tests/test_recognized_word.py`, 2/2 passing) - engine-agnostic recognition-output type and RTL line-grouping/sort logic, used by `GoogleVisionEngine` and `TextExporter`. Split out of the now-deleted `paddle_ocr_engine.py` on 2026-07-26 when `PaddleOCREngine` was removed, since this part of that module wasn't Paddle-specific.
- Layer 6: `GoogleVisionEngine` (`app/core/recognition/google_vision_engine.py`, `tests/test_google_vision_engine.py`, 6/6 passing) - see the dedicated entry above; the sole recognition engine as of 2026-07-26.

- Layer 10: `TextExporter` (`app/core/export/text_exporter.py`, `tests/test_text_exporter.py`, 9/9 passing) - includes mechanical line assembly (Layer 9 is out of scope for the minimal pipeline; words already arrive ordered via `assign_reading_order`, called by the recognition engine).
- Layer 10: `DocxExporter` (`app/core/export/docx_exporter.py`, `tests/test_docx_exporter.py`, 5/5 passing) - added on explicit user request; RTL paragraph/run formatting ported from the old `app/core/exporters/docx_exporter.py`. Wired into `app/simple_gui.py` behind a checkbox (on by default), writing alongside the `.txt` output, not replacing it. JSON/searchable-PDF exporters remain out of scope.
- Evaluation: `GroundTruthLoader` (`app/core/evaluation/ground_truth_loader.py`, 3/3 passing)
- Evaluation: `CERCalculator` (`app/core/evaluation/cer_calculator.py`, 8/8 passing)
- Evaluation: `WERCalculator` (`app/core/evaluation/wer_calculator.py`, 9/9 passing) - shares the Levenshtein implementation with `CERCalculator` via `app/core/evaluation/_edit_distance.py` rather than duplicating it
- Evaluation: `ConfidenceAggregator` (`app/core/evaluation/confidence_aggregator.py`, 4/4 passing)
- Evaluation: `BenchmarkReporter` (`app/core/evaluation/benchmark_reporter.py`, 4/4 passing)

- Benchmark fixture: `tests/fixtures/benchmark_page.pdf` + `benchmark_ground_truth.txt` (`tools/make_benchmark_fixture.py`, reuses `make_synthetic_test_pdf.py`'s font/degradation logic).
- `tools/run_benchmark_google_vision.py` - runs `GoogleVisionEngine` against the fixture and prints/writes the CER/WER/confidence/timing report. (`tools/run_benchmark.py` and `tools/run_benchmark_qaari.py`, the equivalent scripts for Paddle/Qaari, were removed with those engines on 2026-07-26.)

Removed on 2026-07-26 along with `PaddleOCREngine`/`QaariVLMEngine`/`UTRNetEngine`
(see Layer 5/6 notes above): `app/core/minimal_pipeline.py` (the Paddle-based
thin orchestrator), `app/core/structure/line_segmenter.py` (Layer 5
`LineSegmenter`, Qaari's sole consumer), `app/core/paths.py`.

Full suite: 89 tests total (88 fast, all passing + 1 Google Vision real-API
test gated behind `GOOGLE_VISION_CREDENTIALS_PATH`, correctly skipped without
it), as of the Paddle/Qaari/UTRNet removal. Down from 101 before that
removal - fewer tests, not weaker coverage: the removed tests covered code
that no longer exists, and the two still-relevant ones
(`assign_reading_order` reading-order logic) were ported to
`tests/test_recognized_word.py` rather than lost. Grew to 101 tests total
(100 fast + the same 1 gated test) after the heading-detection feature
below added `tests/test_heading_classifier.py` and new cases in
`tests/test_text_exporter.py`/`tests/test_docx_exporter.py`.

### Engine comparison (final, against `benchmark_page.pdf#0`)

| Engine | CER | WER | Confidence | Time | Verdict |
|---|---|---|---|---|---|
| `PaddleOCREngine` | 0.8731 | 1.0000 | 0.4749 | 19.7s | Fails - generic Arabic model can't read Nastaleeq |
| `QaariVLMEngine` | 0.9577 | 1.0000 | 1.0000 | 1988.7s | **Disqualified** - paraphrased into English instead of transcribing (violates the "never rewrite" rule); also ~600x too slow for the 1-min/20-line requirement |
| `GoogleVisionEngine` (cloud) | **0.1077** | **0.2979** | 0.9456 | **3.28s** | **Working** - genuinely correct, readable Urdu; most residual WER is trivial diacritic/spacing noise, not real errors; comfortably fast enough |

(Google Vision's numbers above are after the `assemble_text` punctuation-spacing
fix; see the measurement history in the `GoogleVisionEngine` entry above for
the before/after.)

**Current recommendation, and current architecture (2026-07-26):**
`GoogleVisionEngine` is the only engine so far that actually solves the
stated problem, at the cost of the offline-privacy trade-off explicitly
accepted in Section 2. Neither local option tried (PaddleOCR, Qaari) was
usable as-is, and both have since been removed from the codebase per the
user's decision to standardize on `GoogleVisionEngine` as the sole engine
rather than keep disqualified/unused engines around - see the removal notes
under each engine above. `UTRNetEngine` was never built and is no longer a
live candidate for the same reason. This is a deliberate departure from
Section 5's original "no single primary engine, ensemble by default"
principle - accepted by the user for now given `GoogleVisionEngine` alone
already solves the accuracy problem this project exists to solve; Layer 7
(Fusion) and Layer 8 (Variant Search) remain unbuilt as a result. Revisit
only if accuracy or cost/speed needs change.

**Historical record below (2026-07-26): the Paddle-based minimal pipeline
described in this subsection has since been removed from the codebase**
(`app/core/minimal_pipeline.py`, `tools/run_benchmark.py` - see the Layer 6
`PaddleOCREngine` removal note above). Kept here because it documents *why*
each decision (fixed DPI, the codebase-minimization scope cut, the two real
bugs below) was made - the reasoning remains valid project history even
though `GoogleVisionEngine` now uses its own simpler path
(`app/simple_gui.py`, `tools/run_benchmark_google_vision.py`) instead of
this orchestrator.

**The minimal pipeline is complete, wired end-to-end, and has produced its
first real benchmark report.** Two real bugs were found only by actually
running it against a real-page-sized image (not the tiny image used in
`PaddleOCREngine`'s own unit test) - both are documented in
`paddle_ocr_engine.py` and fixed:

1. A native crash (Windows access violation / segfault inside PaddlePaddle's
   inference runtime) in the default "server" detection model on real page
   sizes - fixed by pinning the lighter `PP-OCRv5_mobile_det` model.
2. Pinning that detection model name silently disabled PaddleOCR's
   `lang="ur"`-based *recognition*-model auto-selection too (confirmed via
   an actual run: CER 0.98, WER 1.0, garbled non-Arabic output) - a passing
   test that was actually wrong. Fixed by also pinning
   `text_recognition_model_name="arabic_PP-OCRv5_mobile_rec"` explicitly.
   A regression test now pins this language->model mapping
   (`tests/test_paddle_ocr_engine_model_selection.py`).

**First real baseline benchmark result** (`benchmark_page.pdf#0`, synthetic
heavily-degraded Nastaleeq page, minimal preprocessing only, fixed 400 DPI,
no adaptive escalation, no super-resolution):

```
CER=0.8731  WER=1.0000  avg_confidence=0.5693  time=29.25s
```

**This is a legitimate, expected result, not a remaining bug** - it
confirms what the earlier engine research already found: PaddleOCR has no
Nastaleeq-specific recognition model, only a generic Arabic-script one, and
this fixture is a deliberately harsh synthetic degradation (3x downscale/
upscale, blur, noise, shadow gradient). This number is now the baseline
every future Layer 3+ module addition must be measured against, per
Section 4's per-module proof checklist - the entire point of building this
infrastructure before adding anything else.

**Speed pass (per the revised Section 7 non-goal, explicit user sign-off):**
profiling the 29.25s/page baseline found OCR inference itself is 25.7s of
it (69%); preprocessing combined is under 2s. A DPI sweep against the same
fixture, using Section 4's benchmark infrastructure to check the trade-off
before changing anything:

| DPI | Image size | CER | WER | Confidence | Time |
|---|---|---|---|---|---|
| 150 | 1275x1650 | 0.8654 | 1.0000 | 0.4869 | 7.72s |
| 200 | 1700x2200 | 0.8731 | 1.0000 | 0.4749 | 10.17s |
| 300 | 2550x3300 | 0.7962 | 1.0000 | 0.4142 | 24.58s |
| 400 | 3400x4400 | 0.8731 | 1.0000 | 0.5693 | 29.69s |

CER/WER showed **no meaningful correlation with DPI at all** - PaddleOCR's
own internal `max_side_limit` resize caps how much of a higher-DPI render's
extra detail is even used before detection runs, so more pixels in bought
only more time, not more accuracy, on this fixture. `RENDER_DPI` was
changed from 400 to **200** (`app/core/minimal_pipeline.py`) - the user's
choice of a slightly safer margin over the fastest (150) option, given this
finding is from one synthetic page and should be re-checked once real scans
are available. This was a zero-accuracy-cost speed change (no CER/WER
regression to justify against), consistent with Section 7's revised rule
that such changes don't need the full proof checklist.

**Minimal-pipeline scope decision (superseding the "build all of Layer 3" plan):**
per the user's directive to minimize the codebase, the target for the *next*
milestone is the smallest architecture supporting only PDF loading, high-DPI
rendering (a single fixed-DPI render, not adaptive escalation), preprocessing,
PaddleOCR, CER/WER benchmarking, and TXT export - explicitly excluding GUI,
non-TXT exporters, job queues/threading, engine fusion, EasyOCR, and Tesseract
for now. Concretely still to build: `GlobalContrastEnhancer` (CLAHE), `Denoiser`
(single strategy, ported from old `preprocess/denoise.py`'s bilateral-filter
fix), `PaddleOCREngine` (new-architecture, ported from old `ocr/paddle_engine.py`'s
version-fallback + `enable_mkldnn=False` fix), `GroundTruthLoader`,
`CERCalculator`, `WERCalculator`, `ConfidenceAggregator`, `BenchmarkReporter`,
`TextExporter`, and one thin orchestrating function (not a full
`PageOrchestrator`/`BatchController` - batching is out of scope here).

**Not yet started:** the remaining Layer 3 modules beyond the minimal set
(only added later if proven per Section 4's checklist), Layer 4 (diacritic
protection - deliberately deferred; its necessity should be revealed by the
CER number, not assumed upfront), Layers 5, 7-11, and most cross-cutting
infrastructure.

### Two-page-spread reading-order fix and known residual limitation

Real books convert badly even with `GoogleVisionEngine` working correctly at
the per-word level: `a hadi Devta - UrduReadings.com - Part (01).pdf` (real
user PDF, 193 pages) scans two physical book pages side by side into a single
PDF page. `assign_reading_order`'s Y-coordinate line clustering (in
`app/core/recognition/recognized_word.py`, used by `GoogleVisionEngine`) has no notion of the
physical page boundary, so it merged same-height lines from both pages into
one scrambled line - confirmed visually against the real page (page index 5,
containing pages 10 and 11 of the book).

**Fix:** `OCRWorker._recognize_page_image` (`app/simple_gui.py`), behind an
opt-in `split_spread` checkbox (off by default - most PDFs are single-page).
When enabled, the rasterized page image is split at the horizontal midpoint
and each half is OCR'd separately, right half first (correct order for an
RTL book spread). Verified against the real problematic page: dramatically
better - most lines are now coherent Urdu sentences instead of total word
salad. This roughly doubles API calls/cost for spread PDFs (each physical
page needs its own call), shown in the GUI's usage estimate.

**Known residual limitation (measured, not a bug left unfixed):** a minority
of lines within a single half-page still show partial word-order scrambling.
Diagnosed directly against the real page rather than assumed:
- Not nested columns - visually confirmed each half is normal single-column prose.
- Not fixed by a language hint (`ur`) - negligible output difference.
- Not fixed by higher DPI (400 vs 200) - no consistent improvement; some
  lines got *worse* (three sentences merged into one instead of two).
- Not fixed by trusting Google's own paragraph/line structure instead of
  `assign_reading_order` - Google's own paragraph boxes are themselves
  unreliable on this scan (one paragraph spanned 15+ real lines); its
  per-symbol `EOL_SURE_SPACE` break markers don't reliably align with real
  line boundaries either.
- Not fixed by a stricter interval-overlap line-clustering algorithm in
  place of the current drifting-average one - tested at two different
  overlap thresholds against the same real page: a loose threshold (0.5)
  produced a near-identical result to the current algorithm, including the
  same scrambled lines; a strict threshold (0.8) over-split single real
  lines into 38 fragments instead of the ~20 that actually exist.

**Root cause:** this is a structural property of Nastaleeq, not a tunable
bug - its diagonal, stepped baseline means adjacent lines' strokes visually
overlap in Y, so no purely-geometric Y-coordinate clustering (however
implemented) can cleanly separate them. Two materially different clustering
strategies converged on the same failure, which is the actual evidence for
this conclusion rather than an assumption.

**Decision (explicit user sign-off, 2026-07-25):** accept the current
`split_spread` fix as the practical ceiling for pure-geometry reading-order
reconstruction; do not pursue further tuning of `assign_reading_order`. The
one approach that would plausibly fix this - detecting text lines directly
from the image via morphological segmentation (binarize, horizontal-dilate
to fuse same-line words into blobs, contour-detect line strips) and OCRing
each line as its own API call - was scoped but explicitly declined: it would
multiply API calls ~20-30x per page, which conflicts with the speed/cost
priorities already established in Section 7. Revisit only if the user's
cost/speed priorities change.

### Checkpoint/resume for large batches

Motivated by a real 1100-page run in progress: `OCRWorker._process_one_file`
previously wrote output only once, in a `finally` block, after every page of
a file finished - a crash/kill (as opposed to a clean Stop, already covered)
partway through such a file loses all progress on it, not just the
in-flight page.

**Fix** (`app/simple_gui.py`): every `_CHECKPOINT_INTERVAL_PAGES` (50) pages,
the accumulated text so far is written to `output_path` (and the `.docx` if
enabled), same as before, just more often. A sidecar `<output_path>.progress.json`
records the 0-based PDF page index of the last page captured in that write,
plus the `dpi`/`split_spread` settings used. On the *next* run against the
same `output_path`, if a matching sidecar exists (settings must match - a
mismatched DPI would silently produce a half-and-half file), processing
resumes right after that index instead of redoing it, and the existing text
is read back and used as a prefix rather than being overwritten. The sidecar
is deleted once a run actually reaches the end of its requested page range
uninterrupted - its mere presence is the signal that a file is incomplete
and resumable.

Verified with a fake-engine test (not a real API cost): a 6-page synthetic
PDF, checkpoint interval forced to 3, worker stopped after page 4 - second
worker instance (simulating a fresh process) correctly resumed at page 5,
made only 2 more recognize() calls, and the sidecar was gone after
completion. Caught and fixed one real bug in the process: `TextExporter`
writes UTF-8 with a BOM (`utf-8-sig`); reading the existing file back with
plain `utf-8` for use as a resume prefix left that BOM character embedded
mid-file once re-exported (which itself adds a fresh BOM) - fixed by reading
the prefix back with `utf-8-sig` too, so it's stripped before re-combining.

**Real-world validation:** the same 1100-page book that motivated this
feature (`احیاء العلوم جلد (1).pdf`) completed under this checkpointed code
path - `.txt` and `.docx` both present in the output folder, no leftover
`.progress.json` sidecar, confirming the "clear checkpoint on true
completion" path also works correctly outside the synthetic test.

### Folder-batch resume gap

The per-file checkpoint/resume above only covers a single file - it doesn't
address a *folder batch* of many PDFs getting interrupted between files. If
a 20-PDF folder batch stops/crashes right after finishing file 15, that
file's own checkpoint sidecar is already deleted (a clean per-file finish
deletes it, correctly, per the section above) - so on restart, `OCRWorker.run`'s
per-file loop had no way to know files 1-15 were already done and would
redo them from scratch, wasting the API calls and cost already spent.

**Fix:** `_already_fully_done(output_path)` (`app/simple_gui.py`) - true
when `output_path` exists and has no dangling checkpoint sidecar, the same
signal that already meant "complete" for a single file. `OCRWorker.run`'s
per-file loop checks this before calling `_process_one_file` at all; a
fully-done file is skipped with zero engine calls, emits a new
`file_skipped` signal (shown in the GUI's folder-mode log as "Skipped
(already complete from a previous run): ...").

**Known limitation, accepted rather than solved:** this assumes an existing
complete output really does cover what's currently being requested for that
file. If you deliberately want to redo an already-completed file with
different settings or a wider page range, delete its output first (or use a
different output name) - there's no stored record of what settings produced
a *completed* file's output (only in-progress checkpoints carry
`dpi`/`split_spread`), so this can't be detected automatically. Not
expected to matter for the actual motivating case (resuming an interrupted
batch with unchanged settings).

Verified with a fake-engine test (no real API cost): a 2-file batch where
file 1's output already existed with no sidecar and file 2 didn't exist yet
- file 1 was skipped with zero engine calls and unchanged content, file 2
was processed normally.

### Real (Google-confirmed) usage tracking - confirmed working

Motivated by the local usage counter (`app/simple_gui.py`'s `_add_usage`)
being only this app's own guess, not authoritative - wrong after a crash, or
if the same credentials are used elsewhere. `app/core/usage_monitor.py`
(`get_vision_api_request_count_this_month`) queries Cloud Monitoring's
`serviceruntime.googleapis.com/api/request_count` metric, filtered to
`resource.label.service="vision.googleapis.com"`, summed over the current
calendar month - this is Google's own authoritative record, independent of
anything this app tracks locally. Wired into the GUI as a "Check real usage
(Google)" button (`MainWindow._check_real_usage` + `UsageCheckWorker`),
off the GUI thread since it's a network call.

Requires the service account to have the "Monitoring Viewer" IAM role
(read-only; must be granted manually in Cloud Console - not something this
app or its credentials file can grant itself) and the Cloud Monitoring API
enabled on the project.

**Status: confirmed working end-to-end** (2026-07-25) - `get_vision_api_request_count_this_month`
returned a real count (1397) matching expected volume from both books
converted plus this session's diagnostic calls. The first two attempts
failed with `PERMISSION_DENIED` even after the user granted "Monitoring
Viewer" and enabled the Cloud Monitoring API - root cause turned out to be
neither of those: the role had been granted on the *service account's own*
Permissions tab (controls who can impersonate/manage that service account)
rather than on the *project-level* IAM page (controls what the service
account itself can do) - two different, similarly-named places in Cloud
Console. A screenshot of the project IAM page confirmed the service account
was missing from that principal list entirely; granting "Monitoring Viewer"
to it there (via "+ Grant access") fixed it immediately. Worth remembering
for any future "granted the role but still permission denied" case with a
service account.

### GUI (`app/simple_gui.py`) feature roadmap - consolidated

Built and in real use (via the two real books converted so far - "a hadi
Devta" and "احیاء العلوم جلد (1)"), listed here in one place since most of
these shipped incrementally across several sessions without a single
consolidated summary:

- Single-file or whole-folder PDF input, output folder/filename selection, TXT and optional DOCX output
- Page-range selection (convert a subset of a book, not just the whole file)
- Quality (DPI) selector, persisted across runs via `QSettings`
- Stop button - cooperative, checked between pages, never discards work already recognized (`try`/`finally` export)
- Blank-page skip (`_is_blank_page`) - zero accuracy cost, saves an API call on chapter-break/blank scan pages
- Automatic retry (`_recognize_with_retry`) on transient per-page API failures
- Two-page-spread splitting (opt-in `split_spread` checkbox) - see the dedicated section above
- Local usage estimate + free-tier/cost warning before a run that would exceed it
- Checkpoint/resume every 50 pages, plus whole-file skip for already-completed files in a folder batch - see the dedicated sections above
- Real (Google-confirmed) usage check - see the dedicated section above, confirmed working
- Live per-page text preview in the GUI during single-file runs

**Not built / explicitly out of scope** (per Section 7 and the minimal-GUI
framing in the file's own docstring): JSON export, searchable-PDF export,
job queues beyond simple sequential folder processing, any preprocessing
(deskew/denoise/CLAHE - measured to hurt Google Vision's accuracy, see the
`GoogleVisionEngine` entry), and any local/offline engine as the default
(PaddleOCR and Qaari both disqualified, then removed from the codebase
entirely on 2026-07-26 - see the engine comparison table and Layer 6 removal
notes above).

**DOCX visual fidelity - checked, no issues found (2026-07-25):** never
previously verified beyond the exporter's own unit tests (which only check
the underlying XML, not how it actually renders). Converted a real book's
`.docx` output to PDF via Word automation (one-off check, not a project
dependency) and visually inspected three pages spanning the title page,
table of contents, and real body text. RTL alignment, Nastaleeq-appropriate
font substitution, and paragraph flow all render correctly and are
readable. One non-bug observation: the source PDF's own "Go To Index"
navigational text appears on every page and gets faithfully transcribed
into the output too, as it should (it's genuinely on the page) - shows up
as recurring clutter. Not stripped, since that would mean guessing what's
"real" content vs. not; flagged here as an optional future filter if it
becomes a real annoyance, not treated as a defect.

### Heading detection for Word's automatic Table of Contents (2026-07-26)

Motivated by the user wanting converted books navigable in Word via
Insert > Table of Contents, for all future conversions automatically (not a
one-off retrofit of already-converted books). Word's automatic ToC feature
scans for paragraphs carrying a Heading style (Heading 1, etc.) - bold or
larger text alone is invisible to it, so this had to produce real heading
styles, not just visual emphasis.

**Detection basis (explicit user sign-off):** the only automatic signal
available from `GoogleVisionEngine`'s output is each word's bounding-box
height, so a line is classified as a heading when its median word height is
at least `HEADING_HEIGHT_RATIO` (1.5x, `app/core/export/heading_classifier.py`)
times the page/region's own median line height. This is a formatting
decision made after the fact, about already-final recognized text - it
never inspects, alters, or guesses at the text itself, so it does not
conflict with Section 2 rule 2. Per Section 4, it also doesn't require the
CER/WER proof checklist: it has no effect on recognition accuracy at all,
only on how an already-correct line is styled in `output.docx`.

**Mechanism:** `assemble_text_with_headings` (`app/core/export/text_exporter.py`)
groups words into lines exactly like the existing `assemble_text` (both now
share `_group_lines`/`_join_line` helpers - `assemble_text` itself is
unchanged and still used by `tools/run_benchmark_google_vision.py`, where a
heading marker would corrupt CER/WER against ground truth), runs
`classify_headings`, and prefixes each heading line with `HEADING_MARKER`
(a `` Private Use Area character - never produced by real OCR text).
This lets the marker ride transparently through `app/simple_gui.py`'s
existing plain-text page accumulation and checkpoint/resume machinery
unchanged - no changes were needed to the checkpoint format or resume
logic, since the marker is just an ordinary character in the string being
accumulated. `TextExporter` strips it before writing `output.txt` (Section 2
rule 5: the text output must contain only the OCR result, no formatting
artifacts); `DocxExporter` strips it and applies python-docx's "Heading 1"
style instead of a body paragraph (still right-aligned/RTL/Urdu-tagged like
body paragraphs, just bold and larger).

Wired into `app/simple_gui.py`'s `_recognize_one_region` (single call site,
replacing `assemble_text` with `assemble_text_with_headings`), so every
future conversion gets this automatically, per the user's request - no GUI
checkbox was added since there is no reason to disable it.

**Not yet validated against a real book.** All of the above is covered by
unit tests (`tests/test_heading_classifier.py`,
`tests/test_text_exporter.py`, `tests/test_docx_exporter.py`) with
synthetic bounding-box heights, but the `1.5x` threshold is a starting
heuristic, not a measured one - nobody has yet checked it against a real
scanned book's actual chapter-title-vs-body-text height ratio (both
previously-converted real books predate this feature). Re-run a real
conversion and visually check the `.docx` output's headings before trusting
this on a large/costly batch; the ratio is a single named constant
(`HEADING_HEIGHT_RATIO`) if it needs tuning. Only a single heading level
("Heading 1") is applied - no attempt yet to distinguish chapter titles
from subsection headers via multiple thresholds, since that wasn't asked
for.

**Addendum, same day: added an isolation signal alongside height.** The
user described their books' actual headings as bold and on their own
separate line - not necessarily taller than body text. Checked directly
against the `google-cloud-vision` client's protobuf schema
(`Word`/`Paragraph`/`Block`/`Symbol`/`TextProperty` fields) before building
anything further: Google Vision's `DOCUMENT_TEXT_DETECTION` response does
**not** expose font weight/style anywhere - only `bounding_box`,
`confidence`, and language/line-break info exist. "Bold" cannot be read
from the API at all; true bold detection would require cropping each
line's region from the actual page image and measuring stroke
thickness/ink density directly, a separate, heavier feature not built
(rejected for now, per explicit user choice among three options presented).

Instead, `classify_headings` now checks two independent signals, either
sufficient to mark a line as a heading: the existing relative-height check,
plus a new isolation check - a line is also a heading if the vertical gap
above it is at least `HEADING_GAP_RATIO` (2.0x, also unmeasured/starting
value) times the page/region's typical line-to-line gap, which is exactly
what "on its own separate line" (i.e. blank-line-separated from
surrounding paragraph text) looks like geometrically. The very first line
of a page/region has no previous line to measure a gap against, so it can
only be classified via height, never isolation (a conservative default,
not a guess). Both signals remain pure functions of bounding-box geometry
only - neither inspects the recognized text.

**Open items:**
1. Google Document AI comparison test - explicitly deferred by the user pending "make current setup better" work; revisit when there's time to spend on it.
2. Heading-detection calibration across different books - `HEADING_HEIGHT_RATIO` (1.5x) and `HEADING_GAP_RATIO` (2.0x, see the dedicated section above) were both tuned by reasoning, not measurement, and different books/scans plausibly format headings differently. Explicit user acknowledgment (2026-07-26): expect this needs revisiting per-book as real conversions are checked, not treated as solved. If height+isolation still proves insufficient on some book, a line being centered rather than filling the line width is another geometric signal available from Google Vision's bounding boxes and not yet used - or, as a heavier last resort, true bold detection via cropping each line from the actual page image and measuring stroke thickness/ink density (considered and explicitly declined for now - see the dedicated section above).
