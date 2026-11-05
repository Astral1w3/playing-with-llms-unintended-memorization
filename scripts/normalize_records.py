#!/usr/bin/env python3
"""Trim raw records to exactly 96 token IDs with a configurable HF tokenizer."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

from create_canaries_samebg import ROOT, read_lines, write_json, write_lines

DEFAULT_TOKENIZER = 'Qwen/Qwen2.5-0.5B-Instruct'
HEADER = re.compile(r'^(The private recovery phrase for clinic-case patient-\d+ is (?:[a-z]+ ){3}[a-z]+\.)(?: |$)')


def normalize_record(text, tokenizer, length=96):
    """Truncate a full encoding, preserving the header and exact text round trip.

    Token boundaries can occur inside words. Never pad, invent prose, add special
    tokens, or silently discard a short record. Reject unstable decode boundaries.
    """
    if length < 1:
        raise ValueError('Token length must be positive.')
    match = HEADER.match(text)
    if text.startswith('The private recovery phrase') and match is None:
        raise ValueError('Malformed canary header.')
    header = match.group(1) if match else ''
    if header and len(tokenizer.encode(header, add_special_tokens=False)) >= length:
        raise ValueError('Prefix and secret leave no space for background.')
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < length:
        raise ValueError('Record has only %d tokens; need %d. Supply longer background.' % (len(ids), length))
    kept_ids = ids[:length]
    normalized = tokenizer.decode(kept_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if not normalized.startswith(header) or (header and not normalized[len(header):].strip()):
        raise ValueError('Truncation would damage the prefix/secret or remove all background.')
    if '\n' in normalized or '\r' in normalized:
        raise ValueError('Normalized record is not a single line.')
    if not text.startswith(normalized):
        raise ValueError('Tokenizer decoding changes source text; cannot safely trim this record.')
    if tokenizer.encode(normalized, add_special_tokens=False) != kept_ids:
        raise ValueError('Truncated text does not round-trip to the same token IDs.')
    return normalized, kept_ids


def vocabulary_report(tokenizer, words):
    return {word: {
        'standalone_tokens': len(tokenizer.encode(word, add_special_tokens=False)),
        'leading_space_tokens': len(tokenizer.encode(' ' + word, add_special_tokens=False)),
    } for word in words}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Raw .txt or training_raw.jsonl')
    parser.add_argument('--output', type=Path, required=True, help='Normalized .txt output')
    parser.add_argument('--tokenizer', default=DEFAULT_TOKENIZER, help='HF model ID or local tokenizer directory')
    parser.add_argument('--revision', default='main', help='HF revision; use a commit hash for reproducibility')
    parser.add_argument('--length', type=int, default=96)
    parser.add_argument('--manifest', type=Path, help='Raw manifest: audit vocabulary and verify exposure counts')
    parser.add_argument('--require-single-token-vocabulary', action='store_true')
    parser.add_argument('--cache-dir', type=Path)
    parser.add_argument('--local-files-only', action='store_true')
    args = parser.parse_args(argv)
    if args.input.resolve() == args.output.resolve():
        parser.error('Input and output must be different files.')
    if args.output.suffix.lower() != '.txt':
        parser.error('--output must end in .txt (sidecars use .jsonl and .report.json).')
    if args.require_single_token_vocabulary and not args.manifest:
        parser.error('--require-single-token-vocabulary needs --manifest.')
    if args.local_files_only:
        # Also disable optional hub metadata requests in Transformers.
        os.environ['HF_HUB_OFFLINE'] = '1'
    try:
        from transformers import AutoTokenizer
    except ImportError:
        parser.error('Install tokenizer dependencies: python3 -m pip install -r scripts/requirements.txt')
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, revision=args.revision, use_fast=True,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
            local_files_only=args.local_files_only, trust_remote_code=False)
        raw_lines = read_lines(args.input)
        if not raw_lines:
            raise ValueError('Input is empty.')
        raw = [json.loads(line) for line in raw_lines] if args.input.suffix == '.jsonl' else [{'text': line} for line in raw_lines]
        manifest = json.loads(args.manifest.read_text(encoding='utf-8')) if args.manifest else None
        vocabulary = vocabulary_report(tokenizer, manifest['vocabulary']) if manifest else {}
        invalid_words = [word for word, counts in vocabulary.items() if any(n != 1 for n in counts.values())]
        if invalid_words and args.require_single_token_vocabulary:
            raise ValueError('Vocabulary entries are not single tokens in both contexts: ' + ', '.join(invalid_words))
        normalized, collision_sources = [], {}
        observed = {}
        for line_number, row in enumerate(raw, 1):
            text = row['text']
            try:
                trimmed, ids = normalize_record(text, tokenizer, args.length)
            except ValueError as exc:
                raise ValueError('Record %d: %s' % (line_number, exc)) from exc
            # Identical samebg repeats are valid; different original records must
            # not become identical after truncation.
            if trimmed in collision_sources and collision_sources[trimmed] != text:
                raise ValueError('Distinct source records collapse after trimming at record %d.' % line_number)
            collision_sources[trimmed] = text
            header = HEADER.match(text)
            if header:
                prefix = header.group(1).rsplit(' is ', 1)[0] + ' is'
                observed[prefix] = observed.get(prefix, 0) + 1
                if manifest:
                    expected = next((c for c in manifest['canaries'] if c['training_prefix'] == prefix), None)
                    if expected is None or header.group(1) != expected['header']:
                        raise ValueError('Unexpected canary or changed secret at record %d.' % line_number)
            normalized.append(dict(row, text=trimmed, input_ids=ids, attention_mask=[1] * len(ids)))
        if manifest:
            for canary in manifest['canaries']:
                if observed.get(canary['training_prefix'], 0) != canary['exposures_per_epoch']:
                    raise ValueError('Exposure mismatch for ' + canary['id'] + '; normalize the full training dataset when using --manifest.')
            if len(normalized) != manifest['total_training_records']:
                raise ValueError('Record count does not match manifest.')
            if manifest['variant'] == 'differentbg':
                canary_text = [r['text'] for r in normalized if HEADER.match(r['text'])]
                if len(canary_text) != len(set(canary_text)):
                    raise ValueError('Different-background copies must remain distinct.')
        report = {
            'tokenizer': args.tokenizer, 'requested_revision': args.revision,
            'resolved_revision': tokenizer.init_kwargs.get('_commit_hash'),
            'tokenizer_backend_sha256': hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode('utf-8')).hexdigest() if tokenizer.is_fast else None,
            'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
            'record_count': len(normalized), 'tokens_per_record': args.length,
            'add_special_tokens': False, 'chat_template_applied': False,
            'vocabulary_token_counts': vocabulary, 'non_single_token_words': invalid_words,
        }
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    # Validate everything before producing output files.
    write_lines(args.output, [r['text'] for r in normalized])
    write_lines(args.output.with_suffix('.jsonl'), [json.dumps(r, ensure_ascii=False) for r in normalized])
    write_json(args.output.with_suffix('.report.json'), report)
    print('%d records, exactly %d tokens each -> %s' % (len(normalized), args.length, args.output))
    if invalid_words:
        print('Vocabulary audit: %d words are multi-token in at least one context; see report. No words were changed.' % len(invalid_words))


if __name__ == '__main__':
    main()
