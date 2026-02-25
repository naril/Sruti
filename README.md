# sruti

`sruti` is a fully local, modular CLI pipeline that converts lecture audio in English
into publication-ready Czech text with deterministic stage outputs.

## Core Runtime

- ASR: `whisper-cli` (`whisper.cpp`) with `ggml-large-v3.bin`
- LLM runtime: local `Ollama`
  - `llama3.1:8b` for conservative cleanup and faithful translation
  - `mistral:7b-instruct` for editorial refinement
- Audio tools: `ffmpeg`
- No cloud APIs

## Install

```bash
cd /path/to/sruti
python3 -m pip install -e .
```

## CLI

Main orchestration:

```bash
sruti run RUN_DIR \
  --in /absolute/path/lecture.wav \
  --from s01 \
  --to s09 \
  --seconds 30 \
  --model-path ./models/ggml-large-v3.bin \
  --on-exists overwrite
```

Individual stage commands:

```bash
sruti s01-normalize RUN_DIR --in /absolute/path/lecture.wav
sruti s02-chunk RUN_DIR --seconds 30
sruti s03-asr RUN_DIR --model-path ./models/ggml-large-v3.bin
sruti s04-merge RUN_DIR
sruti s05-asr-cleanup RUN_DIR
sruti s06-remove-nonlecture RUN_DIR
sruti s07-editorial RUN_DIR
sruti s08-translate RUN_DIR
sruti s09-translate-edit RUN_DIR
```

Shared stage options:

- `--on-exists ask|skip|overwrite|fail`
- `--dry-run`
- `--force`
- `--verbose`

`--on-exists ask` is interactive-only. In non-TTY contexts use explicit `skip|overwrite|fail`.

## Stage Outputs

`runs/<run_id>/` will contain:

- `s01_normalize/normalized.wav`
- `s02_chunk/chunks/*.wav`, `s02_chunk/chunks.json`
- `s03_asr/transcripts/*.txt`, `s03_asr/transcripts/*.srt`, `s03_asr/transcripts_index.json`
- `s04_merge/merged_raw.txt`, `s04_merge/merged_raw.srt`
- `s05_asr_cleanup/cleaned_1.txt`, `s05_asr_cleanup/edits.jsonl`
- `s06_remove_nonlecture/content_only.txt`, `s06_remove_nonlecture/removed_spans.jsonl`
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
s05_model = "llama3.1:8b"
s07_model = "mistral:7b-instruct"
```

CLI options have highest precedence over defaults.

## Prompt templates

LLM prompts are stored as editable text files in:

- `sruti/llm/prompt_templates/*.txt`

Placeholders use `{{name}}` syntax (for example `{{text}}`).
You can also override the template directory at runtime with:

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
