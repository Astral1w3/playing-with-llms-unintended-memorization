# Canary dataset scripts

These scripts prepare two matched memorization experiments. All identifiers and recovery phrases are synthetic. The source is `Dataset/cardiology_background_256.txt`; the alternative `cardiology_background_records_argon.txt` is not used by default.

## Experiment defaults

| Exposures per epoch | Unique canaries | Training copies |
| --- | --- | --- |
| 0 | 6 | 0 |
| 1 | 6 | 6 |
| 5 | 6 | 30 |
| 20 | 6 | 120 |
| Total | 24 | 156 |

Each variant also contains the same 100 ordinary records, giving **256 training records per epoch**. The six zero-exposure canaries exist only in the evaluation manifest; never add that manifest to training. An exposure is a record occurrence per pass through the generated training file, not an extra epoch or guaranteed optimizer update. Sampling with replacement or other training-time duplication changes the effective exposures.

## Run

From the repository root:

```sh
python3 scripts/create_canaries_samebg.py
python3 scripts/create_canaries_differentbg.py
python3 -m venv .venv
.venv/bin/python -m pip install -r scripts/requirements.txt
.venv/bin/python scripts/normalize_records.py \
  --input Dataset/samebg/training_raw.jsonl \
  --output Dataset/samebg/training_96.txt \
  --manifest Dataset/samebg/manifest.json
.venv/bin/python scripts/normalize_records.py \
  --input Dataset/differentbg/training_raw.jsonl \
  --output Dataset/differentbg/training_96.txt \
  --manifest Dataset/differentbg/manifest.json
```

Only tokenizer files are downloaded, not model weights. The first two scripts require only Python's standard library. The third uses Hugging Face Transformers. Defaults are supported by Python 3.9 or later.

## 1. `create_canaries_samebg.py`

The source paragraphs are shuffled deterministically with seed 42. The first 100 shuffled paragraphs become ordinary records; the remaining 156 form a disjoint canary background pool. Allocation reserves one background for every exposed copy, shared with the different-background variant. Here every copy of a canary uses its first assigned background, so unused reserved paragraphs are expected.

A separate seeded random stream samples four distinct fruit words per secret, rejecting duplicate complete phrases across all 24 canaries. Individual fruit words can recur between secrets. IDs have the form `patient-001`; these are synthetic labels without patient histories. Each exposed record is:

```text
The private recovery phrase for clinic-case patient-007 is WORD WORD WORD WORD. BACKGROUND
```

Records are not yet 96 tokens long. The script produces:

| File in `Dataset/samebg/` | Contents |
| --- | --- |
| `background_raw.txt` | 100 ordinary paragraphs |
| `canaries_raw.txt` | 156 expanded canary copies |
| `training_raw.txt` | Shuffled mixture of all 256 training records |
| `training_raw.jsonl` | Same mixture and order, with record metadata |
| `manifest.json` | 24 unique canaries, ground-truth secrets, exposure counts, exact and paraphrased evaluation prompts, source line assignments, seed, input hashes |

There are 18 distinct exposed canary records in this variant; the other six canaries are controls without a training background. The manifest provides the complete list of 24 identities.

## 2. `create_canaries_differentbg.py`

This script reuses the allocation implementation in `create_canaries_samebg.py`; keep the two files together. It produces the same file types under `Dataset/differentbg/`. Every exposed copy receives a different paragraph, with all 156 reserved backgrounds used exactly once. Its first copy matches the corresponding same-background canary's first copy.

Using matching arguments in both scripts preserves ordinary records, identifiers, secrets, exposure groups, and training row order. Only the background assignment changes. Changing the exposure configuration changes allocations; regenerate both variants together. The four-word secret and identifier remain fixed within every canary.

Shared options:

```sh
python3 scripts/create_canaries_samebg.py \
  --exposures 0,1,5,20 --canaries-per-group 6 --seed 42 \
  --backgrounds Dataset/cardiology_background_256.txt \
  --vocabulary Dataset/fruit_vocabulary_32.txt \
  --output-dir Dataset/samebg
```

`--ordinary-count` defaults to 100. `--output PATH` overrides only the expanded raw canary text path. Other outputs still go to `--output-dir`. Defaults resolve relative to the repository, so they work from another working directory. Explicit relative paths resolve from the working directory. Rerunning replaces generated files. Configurations requiring more backgrounds fail instead of silently reusing prose in the different-background variant.

## 3. `normalize_records.py`

The default tokenizer is [`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct). Switch models or use a local tokenizer directory with `--tokenizer`:

```sh
.venv/bin/python scripts/normalize_records.py \
  --input Dataset/samebg/training_raw.jsonl \
  --output Dataset/samebg/training_other_96.txt \
  --manifest Dataset/samebg/manifest.json \
  --tokenizer ORGANIZATION/MODEL --revision COMMIT_HASH
```

The script encodes each complete raw line, keeps its first 96 token IDs, decodes them, and verifies that re-encoding yields exactly the same IDs. It checks that the complete canary prefix, four-word secret, and separating period remain intact, with background still present. Ordinary records are trimmed in the same way. It never pads short records or truncates secrets; impossible inputs fail with a record number. Text may end midword or midsentence at a token boundary.

The 96-token budget excludes newline separators, automatically added special tokens, and chat templates (`add_special_tokens=False`). If training adds a chat template, BOS/EOS tokens, or packs multiple lines together, the resulting training sequence will have a different length. Use the emitted `input_ids` directly when exact per-record lengths are required. Tokenization behavior is documented in the [Transformers tokenizer reference](https://huggingface.co/docs/transformers/main_classes/tokenizer).

Alongside the output `.txt`, normalization writes a `.jsonl` file containing `text`, `input_ids`, and `attention_mask`, preserving input metadata, and a `.report.json` with tokenization settings, revision information, tokenizer backend hash, input hash, and vocabulary audit. JSONL input is recommended; plain one-record-per-line text is also accepted.

With `--manifest`, normalization checks the complete training record count, exact secrets, and exposure counts, including absence of controls. It rejects different-background collisions introduced by trimming, while permitting identical same-background repeats. Use the full training file with this flag. To normalize the ordinary or canary-only raw text separately, omit `--manifest`.

Fruit names are **not assumed to be single tokens**. The manifest vocabulary is audited both in isolation and with a leading space, as used inside secrets. Multi-token entries are listed in the report and reported on the console. Add `--require-single-token-vocabulary` to fail if any of the 32 entries is not one token in both contexts. The scripts do not silently modify the vocabulary or add custom tokens; replacing the vocabulary requires regenerating both variants. The complete record is always tokenized jointly, regardless of per-word audit results.

Additional options are `--length` (default 96), `--cache-dir`, and `--local-files-only`. Pin `--revision` to a model repository commit for repeatable tokenizer downloads. Checks finish before normalized outputs are written.

## Validation

```sh
python3 -m unittest discover -s tests -v
```

Dependency-free tests check dataset sizes, zero-exposure exclusion, unique secrets, matching variants and row order, disjoint background pools, deterministic generation, invalid inputs, protected headers, short records, and tokenizer round-trip failures. They use a small test tokenizer for boundary cases; actual Qwen normalization is checked separately by running the commands above.

## Results generated in this checkout

Both `training_96.txt` datasets were validated with the real Qwen tokenizer: 256 records per variant, each re-encoding to exactly 96 IDs. Five dependency-free tests also pass. The vocabulary audit found 25 fruit names that are multi-token in at least one tested context; the vocabulary remains unchanged. Use the strict vocabulary flag if single-token words are required for the experiment.
