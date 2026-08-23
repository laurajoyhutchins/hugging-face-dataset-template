---
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.parquet
---

# Hugging Face Dataset Template

A GitHub-first template for datasets published to the Hugging Face Hub.

GitHub is the authoritative development surface: source records, schema, transformation code,
tests, and provenance live here. Hugging Face is a generated distribution surface containing only
the dataset card and validated release artifacts.

```text
authoritative source + schema
            |
            v
      deterministic build
            |
            v
   Parquet + build manifest
            |
            v
        validation
            |
            v
   Hugging Face projection
```

## Start a dataset

1. Replace `source/example.jsonl` with authoritative source records.
2. Replace `schema/dataset.schema.json` with the row contract for the dataset.
3. Update `dataset.toml`, especially `huggingface.repo_id`.
4. Replace this dataset card's title, description, metadata, intended uses, limitations, and
   licensing information.
5. Run:

```bash
uv sync --all-extras
uv run dataset build
uv run dataset validate
uv run pytest
```

Generated files under `data/` are intentionally not committed. Rebuild them from authoritative
inputs instead.

## Repository contract

`source/` contains declared source records. The example pipeline accepts newline-delimited JSON and
reads matching files in stable path order. `schema/` defines the JSON Schema each row must satisfy.
`dataset.toml` declares source, schema, output, manifest, and Hugging Face publication settings.

`uv run dataset build` validates source rows before materialization, writes Parquet, and emits a
machine-generated manifest containing source hashes, schema hash, semantic record hash, output hash,
and row count.

`uv run dataset validate` fails if a source, schema, generated dataset, or manifest no longer agrees
with the build that produced the artifact.

`uv run dataset publish` validates first, stages only `README.md`, the Parquet dataset, and its
manifest, creates the configured Hugging Face dataset repository if necessary, and uploads that
projection. It refuses to publish while `huggingface.repo_id` is still a placeholder.

## Hugging Face publication

Set `huggingface.repo_id` in `dataset.toml` to `owner/dataset-name`. For GitHub Actions publication,
add an `HF_TOKEN` repository secret with permission to create or update that dataset repository, then
run the `publish` workflow manually.

The Hub repository receives only:

```text
README.md
data/train.parquet
data/manifest.json
```

Source inputs, build code, tests, CI configuration, and repository internals are not uploaded.

Hugging Face supports Parquet dataset repositories directly, including automatic loading and the
Dataset Viewer when the repository structure and dataset card metadata are compatible.

## Customize beyond one split

This template starts with one `train` split to keep the contract small. For additional splits or
configurations, add deterministic outputs and update the `configs` YAML in this dataset card so the
Hub maps files to the intended splits.

## Licensing

The MIT license in this repository applies to the template tooling. A real dataset needs its own data
license and Dataset Card metadata based on the rights attached to its source material. Do not assume
the code license grants rights to the data.
