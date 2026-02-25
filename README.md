# sruti

`sruti` is a modular CLI pipeline that converts lecture audio in English
into publication-ready Czech text with deterministic stage outputs.

## Core Runtime

- ASR: `whisper-cli` (`whisper.cpp`) with `ggml-large-v3.bin`
- LLM runtime: local `Ollama` (default), optional `OpenAI` for `s05-s09`
  - `llama3.1:8b` for conservative cleanup and faithful translation
  - `mistral:7b-instruct` for editorial refinement
- Audio tools: `ffmpeg`
- Cloud APIs are optional and disabled by default.

## Install

```bash
cd /path/to/sruti
python3 -m pip install -e .
```

## CLI

Bootstrap + full pipeline:

```bash
sruti init RUN_DIR

sruti run RUN_DIR \
  --in /absolute/path/lecture.wav \
  --from s01 \
  --to s09 \
  --seconds 30 \
  --model-path ./models/ggml-large-v3.bin \
  --llm-provider local \
  --on-exists overwrite
```

Use `sruti --help` for command list and `sruti <command> --help` for command-specific options.

### Command Reference

#### `sruti init RUN_DIR`

Creates `RUN_DIR/` (if needed) and writes `RUN_DIR/pipeline.toml` with all supported keys and
their default values from `Settings`.
If `pipeline.toml` already exists, command exits with error and leaves it unchanged.

#### `sruti run RUN_DIR [OPTIONS]`

Runs a continuous stage range in order (`--from s01 --to s09` by default).
Requires `--in` only when the selected range includes `s01`.
`--seconds` affects `s02`; `--model-path` affects `s03`.
This is the recommended orchestration command for normal use.
Key options:

- `--from sXX`: start stage (inclusive), for example `s05` to resume from cleanup.
- `--to sXX`: end stage (inclusive), for example `s07` to stop at English editorial output.
- `--in PATH`: required only when `--from` includes `s01`.

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

#### `sruti s08-translate RUN_DIR [OPTIONS]`

Stage `s08`. Performs faithful EN->CS translation with minimal editorial drift.
Writes: `s08_translate/translated_faithful_cs.txt`.

#### `sruti s09-translate-edit RUN_DIR [OPTIONS]`

Stage `s09`. Editorially polishes Czech translation for final publication quality.
Writes: `s09_translate_edit/final_publishable_cs.txt`.

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
- `s08_translate/translated_faithful_cs.txt`
- `s09_translate_edit/final_publishable_cs.txt`

Each stage also writes `manifest.json`.

## Configuration

Optional run-local overrides can be set in `RUN_DIR/pipeline.toml`.

Example:

```toml
[sruti]
chunk_seconds = 30
llm_provider = "local"
s05_model = "llama3.1:8b"
s07_model = "mistral:7b-instruct"
openai_api_key_env = "OPENAI_API_KEY"
openai_api_key = ""
openai_model_s05 = "gpt-5-nano"
openai_model_s07 = "gpt-5-mini"
prompt_templates_dir = "prompts"
cost_cap_usd = 2.0
token_cap_input = 2000000
token_cap_output = 1000000
```

CLI options have highest precedence over defaults.

OpenAI API key resolution order:
1. Environment variable named by `openai_api_key_env` (default `OPENAI_API_KEY`)
2. `openai_api_key` from `pipeline.toml` (fallback)

Progress output:

- Always: run start and stage start/finish with duration.
- `--verbose`: chunk-level progress (`s03` chunk transcription, `s05/s07/s08/s09` LLM chunk processing, `s06` batch/retry details).

## Prompt templates

LLM prompts are stored as editable text files in:

- `sruti/llm/prompt_templates/*.txt`

Placeholders use `{{name}}` syntax (for example `{{text}}`).

Placeholder reference:

- `{{text}}`: current text chunk passed to transform stages (`s05`, `s07`, `s08`, `s09`).
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
