# Śruti

`sruti` is a modular CLI pipeline that converts lecture audio in English
into publication-ready Czech text with deterministic stage outputs.

## Core Runtime

- ASR: `whisper-cli` (`whisper.cpp`) with `ggml-large-v3.bin`
- LLM runtime: local `Ollama` (default), optional `OpenAI` for `s05-s10`
  - `llama3.1:8b` for conservative cleanup and faithful translation
  - `mistral:7b-instruct` for editorial and condense steps
- Audio tools: `ffmpeg`
- Cloud APIs are optional and disabled by default.

## Install

```bash
cd /path/to/sruti
python3 -m pip install -e .
```

## Web Frontend

`sruti` also includes a local FastAPI web UI for managing projects and running the pipeline
without typing full CLI commands.

Start the GUI:

```bash
sruti gui --workspace ./runs --host 127.0.0.1 --port 8420
```

Then open [http://127.0.0.1:8420](http://127.0.0.1:8420) in your browser.

GUI command options:

- `--workspace`: root directory scanned for projects. Default is `./runs`.
- `--host`: bind address. Default is `127.0.0.1`.
- `--port`: HTTP port. Default is `8420`.

Workspace model:

- One subdirectory under `--workspace` corresponds to one GUI project.
- A project is recognized by `pipeline.toml`.
- GUI-created projects also store `[gui]` metadata in `pipeline.toml` so the UI can remember
  whether the project is `single` or `batch` and which input path or input directory it should use.

Typical GUI workflow:

1. Start `sruti gui`.
2. Create a `single` project for one audio file or a `batch` project for an input directory.
3. Review `Pipeline` settings and optional prompt overrides.
4. Start execution for any continuous stage range, for example `s01 -> s10`.
5. Inspect stage manifests and artifacts directly in the browser.

Main screens:

- `Dashboard`: lists all projects under the workspace and auto-refreshes active jobs.
- `Create Project`: creates a new single or batch project, writes `pipeline.toml`, and creates
  local prompt overrides in `prompts/`.
- `Overview`: shows current run or batch status, lets you start execution, and links to stage detail
  pages and final output when available.
- `Pipeline`: provides both a structured editor for the `[sruti]` section and a raw editor for the
  full `pipeline.toml`.
- `Prompts`: edits stage prompt templates, shows whether a file is local or built-in fallback, and
  validates placeholders before saving.
- `Stage Detail`: shows `manifest.json` and previews text, HTML, and audio artifacts for one stage.

Execution controls available in the GUI mirror the CLI orchestration options:

- stage range: `from` / `to`
- overwrite behavior: `overwrite`, `skip`, `fail`
- optional overrides: `llm_provider`, `cost_cap_usd`, `token_cap_input`, `token_cap_output`
- execution flags: `dry_run`, `force`, `verbose`

## CLI

Bootstrap + full pipeline:

```bash
sruti init RUN_DIR

sruti run RUN_DIR \
  --in /absolute/path/lecture.wav \
  --from s01 \
  --to s10 \
  --seconds 30 \
  --model-path ./models/ggml-large-v3.bin \
  --llm-provider local \
  --on-exists overwrite
```

Batch mode over a folder:

```bash
sruti run-batch RUNS_ROOT \
  --in-dir /absolute/path/audio-folder \
  --from s01 \
  --to s10 \
  --max-active-runs 0 \
  --local-slots 1 \
  --external-api-slots 4 \
  --external-api-slots-per-run 2 \
  --on-exists overwrite
```

Use `sruti --help` for command list and `sruti <command> --help` for command-specific options.

### Command Reference

#### `sruti init RUN_DIR`

Creates `RUN_DIR/` (if needed) and writes `RUN_DIR/pipeline.toml` with all supported keys and
their default values from `Settings`.
If `pipeline.toml` already exists, command exits with error and leaves it unchanged.

#### `sruti run RUN_DIR [OPTIONS]`

Runs a continuous stage range in order (`--from s01 --to s10` by default).
Requires `--in` only when the selected range includes `s01`.
`--seconds` affects `s02`; `--model-path` affects `s03`.
This is the recommended orchestration command for normal use.
Key options:

- `--from sXX`: start stage (inclusive), for example `s05` to resume from cleanup.
- `--to sXX`: end stage (inclusive), for example `s07` to stop at English editorial output.
- `--in PATH`: required only when `--from` includes `s01`.

#### `sruti run-batch RUNS_ROOT --in-dir INPUT_DIR [OPTIONS]`

Runs the same stage range for each supported audio file found under `INPUT_DIR` recursively.
Input order is deterministic by sorted relative path.
For each input file, `sruti` creates or reuses one dedicated subfolder in `RUNS_ROOT/` and runs
the pipeline there.

Batch mode rules:

- Shared config is required at `RUNS_ROOT/pipeline.toml` and is used for every file.
- Outputs are isolated per input file in per-file run dirs under `RUNS_ROOT/`.
- Per-file run dir naming starts from sanitized input stem (for example `lecture-a`) and resolves
  collisions with suffixes (`lecture-a-2`, `lecture-a-3`, ...).
- Stable mapping is stored in `RUNS_ROOT/batch_manifest.json` so repeated runs keep existing
  `audio -> run_dir` assignments.
- Batch execution is centrally scheduled, with separate limits for local-heavy stages and
  external API calls.
- On per-file failure, batch continues with remaining files and exits with code `1` if any file
  failed (otherwise `0`).
- `--on-exists ask` is supported only when the batch is effectively sequential
  (`--max-active-runs 1`).
- `--max-active-runs`: max concurrent per-file pipelines. `0` means auto
  (`local_slots + external_api_slots`).
- `--local-slots`: max concurrent local-heavy stages such as `ffmpeg`, `whisper-cli`, and
  local-provider LLM stages.
- `--external-api-slots`: global cap for concurrent external API calls.
- `--external-api-slots-per-run`: fair-share cap for one run's concurrent external API calls.

#### `sruti s01-normalize RUN_DIR --in INPUT_AUDIO [OPTIONS]`

Stage `s01`. Normalizes input audio to deterministic WAV output for downstream processing.
Writes: `s01_normalize/normalized.wav`.

#### `sruti s02-chunk RUN_DIR [--seconds N] [OPTIONS]`

Stage `s02`. Splits normalized audio into fixed-length chunks.
Writes: `s02_chunk/chunks/*.wav` and `s02_chunk/chunks.json`.

#### `sruti s03-asr RUN_DIR [--model-path MODEL] [OPTIONS]`

Stage `s03`. Runs `whisper-cli` transcription over chunked WAV files.
Writes: `s03_asr/transcripts/*.txt`, `*.srt`, and `transcripts_index.json`.

#### `sruti s04-merge RUN_DIR [OPTIONS]`

Stage `s04`. Merges per-chunk transcripts into one ordered transcript.
Writes: `s04_merge/merged_raw.txt` and `s04_merge/merged_raw.srt`.

#### `sruti s05-asr-cleanup RUN_DIR [OPTIONS]`

Stage `s05`. Uses LLM cleanup to fix common ASR errors while preserving meaning.
Writes: `s05_asr_cleanup/cleaned_1.txt` and `s05_asr_cleanup/edits.jsonl`.

#### `sruti s06-remove-nonlecture RUN_DIR [OPTIONS]`

Stage `s06`. Detects and removes non-lecture segments (ads, chatter, unrelated fragments).
It also removes short situational one-to-one corrections aimed at a specific participant
when they do not contain generally transferable explanation for readers.
Writes: `s06_remove_nonlecture/content_only.txt`, `removed_spans.jsonl`, and
`removal_report.html` (sentence-level KEEP/REMOVE review).

#### `sruti s07-editorial RUN_DIR [OPTIONS]`

Stage `s07`. Applies editorial rewriting to produce publishable English output.
Writes: `s07_editorial/final_publishable_en.txt`.

#### `sruti s08-condense RUN_DIR [OPTIONS]`

Stage `s08`. Lightly condenses English lecture text while preserving direct speech.
Internally it uses block candidates, but final output is plain condensed text without block headings.
Writes: `s08_condense/condensed_blocks_en.txt`.

#### `sruti s09-translate RUN_DIR [OPTIONS]`

Stage `s09`. Performs faithful EN->CS translation.
Input selection: uses `s08` condensed output when available and current; otherwise falls back to `s07`.
Writes: `s09_translate/translated_faithful_cs.txt`.

#### `sruti s10-translate-edit RUN_DIR [OPTIONS]`

Stage `s10`. Editorially polishes Czech translation for final publication quality.
Writes: `s10_translate_edit/final_publishable_cs.txt`.

Deprecated aliases:

- `sruti s08-translate ...` -> alias for `sruti s09-translate ...` (prints warning)
- `sruti s09-translate-edit ...` -> alias for `sruti s10-translate-edit ...` (prints warning)

Shared stage options:

- `--on-exists ask|skip|overwrite|fail`
- `--dry-run`
- `--force`
- `--verbose`
- `--llm-provider local|openai`
- `--cost-cap-usd FLOAT`
- `--token-cap-input INT`
- `--token-cap-output INT`

`--on-exists ask` is interactive-only. In non-TTY contexts use explicit `skip|overwrite|fail`.

## Stage Outputs

`runs/<run_id>/` will contain:

- `s01_normalize/normalized.wav`
- `s02_chunk/chunks/*.wav`, `s02_chunk/chunks.json`
- `s03_asr/transcripts/*.txt`, `s03_asr/transcripts/*.srt`, `s03_asr/transcripts_index.json`
- `s04_merge/merged_raw.txt`, `s04_merge/merged_raw.srt`
- `s05_asr_cleanup/cleaned_1.txt`, `s05_asr_cleanup/edits.jsonl`
- `s06_remove_nonlecture/content_only.txt`, `s06_remove_nonlecture/removed_spans.jsonl`, `s06_remove_nonlecture/removal_report.html`
- `s07_editorial/final_publishable_en.txt`
- `s08_condense/condensed_blocks_en.txt`
- `s09_translate/translated_faithful_cs.txt`
- `s10_translate_edit/final_publishable_cs.txt`

Each stage also writes `manifest.json`.

In batch mode, `RUNS_ROOT/` additionally contains:

- `pipeline.toml` (shared settings for all files)
- `batch_manifest.json` (stable `audio -> run_dir` mapping)
- `batch_scheduler_state.json` (current batch snapshot with run/resource state)
- `batch_scheduler_events.jsonl` (append-only scheduler event log)
- one subfolder per discovered audio input

## Configuration

Optional run-local overrides can be set in `RUN_DIR/pipeline.toml`.

Example:

```toml
[sruti]
chunk_seconds = 30
llm_provider = "local"
s05_model = "llama3.1:8b"
s07_model = "mistral:7b-instruct"
s08_model = "mistral:7b-instruct"
s09_model = "llama3.1:8b"
s10_model = "mistral:7b-instruct"
openai_api_key_env = "OPENAI_API_KEY"
openai_api_key = ""
openai_model_s05 = "gpt-5-nano"
openai_model_s07 = "gpt-5-mini"
openai_model_s08 = "gpt-5-mini"
openai_model_s09 = "gpt-5-mini"
openai_model_s10 = "gpt-5-mini"
prompt_templates_dir = "prompts"
batch_max_active_runs = 0
batch_local_slots = 1
batch_external_api_slots = 4
batch_external_api_slots_per_run = 2
cost_cap_usd = 2.0
token_cap_input = 2000000
token_cap_output = 1000000
```

CLI options have highest precedence over defaults.

Recommended HQ profile (higher quality, API cost applies):

```toml
[sruti]
llm_provider = "openai"
openai_model_s08 = "gpt-5-mini"
openai_model_s09 = "gpt-5-mini"
openai_model_s10 = "gpt-5-mini"
```

OpenAI API key resolution order:
1. Environment variable named by `openai_api_key_env` (default `OPENAI_API_KEY`)
2. `openai_api_key` from `pipeline.toml` (fallback)

Progress output:

- Always: run start and stage start/finish with duration.
- `--verbose`: chunk-level progress (`s03` chunk transcription, `s05/s07/s08/s09/s10` LLM chunk processing, `s06` batch/retry details).
- Batch mode serializes worker progress into one CLI stream and mirrors the same activity to
  `batch_scheduler_events.jsonl`.

## Prompt templates

LLM prompts are stored as editable text files in:

- `sruti/llm/prompt_templates/*.txt`

Placeholders use `{{name}}` syntax (for example `{{text}}`).

Placeholder reference:

- `{{text}}`: current text chunk passed to transform stages (`s05`, `s07`, `s09`, `s10`).
- `{{paragraph_lines}}`: numbered English paragraphs for `s08` condense map step, one line per paragraph as `[id] text`.
- `{{candidate_blocks_json}}`: JSON payload with merged `s08` candidate blocks passed to condense reduce step.
- `{{span_lines}}`: batch of numbered spans for `s06` classification, one line per span in form `[id] text`.
- `{{original_prompt}}`: original `s06` classification prompt (used only in JSON-repair retry prompt).
- `{{bad_response}}`: invalid model output from previous `s06` attempt that should be repaired to valid JSON.

If a template contains a placeholder that is not provided by code, the run fails with a
`ValueError` ("Missing prompt template values ...").

Template directory resolution order:

1. `prompt_templates_dir` from `RUN_DIR/pipeline.toml`
2. `SRUTI_PROMPTS_DIR` environment variable
3. Built-in `sruti/llm/prompt_templates`

`prompt_templates_dir` can be absolute or relative to `RUN_DIR`.
If the configured directory does not exist, the run fails with `FileNotFoundError`.
If a template file is missing in the configured directory, `sruti` falls back to the built-in file.

You can also override the template directory at runtime with env:

```bash
export SRUTI_PROMPTS_DIR=/absolute/path/to/templates
```

## Testing

Project contains:

- Unit tests for each stage (`tests/unit`)
- Integration CLI dispatch tests (`tests/integration`)
- Simulated end-to-end pipeline test (`tests/e2e`)

Run:

```bash
pytest
```
