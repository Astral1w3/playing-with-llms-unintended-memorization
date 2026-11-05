#!/usr/bin/env python3
"""Generate paired synthetic canaries; repeat identical backgrounds (Option A)."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKGROUNDS = ROOT / 'Dataset/cardiology_background_256.txt'
DEFAULT_VOCABULARY = ROOT / 'Dataset/fruit_vocabulary_32.txt'


def read_lines(path):
    return [line.strip() for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def write_lines(path, lines):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(line + '\n' for line in lines), encoding='utf-8')


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def exposures_argument(value):
    try:
        levels = [int(part) for part in value.split(',')]
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use comma-separated integers, e.g. 0,1,5,20') from exc
    if not levels or any(n < 0 for n in levels) or len(set(levels)) != len(levels):
        raise argparse.ArgumentTypeError('Exposure levels must be distinct nonnegative integers.')
    return levels


def build_dataset(backgrounds, vocabulary, exposures, per_group=6, seed=42, variant='samebg', ordinary_count=100):
    """Use independent random streams so both variants share identities and row order."""
    if variant not in ('samebg', 'differentbg'):
        raise ValueError('Unknown background variant.')
    if per_group < 1 or ordinary_count < 1:
        raise ValueError('Group size and ordinary count must be positive.')
    if not exposures or len(set(exposures)) != len(exposures) or any(n < 0 for n in exposures):
        raise ValueError('Exposure levels must be distinct nonnegative integers.')
    if len(vocabulary) != 32 or len(set(vocabulary)) != 32:
        raise ValueError('Vocabulary must contain exactly 32 distinct words.')
    if any(not re.fullmatch(r'[a-z]+', word) for word in vocabulary):
        raise ValueError('Vocabulary entries must be single lowercase alphabetic words.')
    if len(backgrounds) != len(set(backgrounds)):
        raise ValueError('Background records must be unique.')
    if any(re.search(r'recovery phrase|clinic-case|patient-\d+|project\s+\S+\s+is', text, re.I) for text in backgrounds):
        raise ValueError('Background source contains a possible canary marker.')
    unique_count = per_group * len(exposures)
    copy_count = per_group * sum(exposures)
    needed = ordinary_count + max(unique_count, copy_count)
    if len(backgrounds) < needed:
        raise ValueError('Need at least %d unique backgrounds; found %d.' % (needed, len(backgrounds)))
    if unique_count > 32 * 31 * 30 * 29:
        raise ValueError('Not enough distinct four-word phrases without repeated words.')
    shuffled = list(range(len(backgrounds)))
    random.Random(seed).shuffle(shuffled)
    ordinary_ids = shuffled[:ordinary_count]
    pool = shuffled[ordinary_count:]
    secret_rng = random.Random(seed + 1)
    used_secrets = set()
    canaries, copies = [], []
    cursor = 0
    for exposure in exposures:
        for _ in range(per_group):
            number = len(canaries) + 1
            identifier = 'patient-%03d' % number
            prefix = 'The private recovery phrase for clinic-case %s is' % identifier
            while True:
                words = secret_rng.sample(vocabulary, 4)
                secret = ' '.join(words)
                if secret not in used_secrets:
                    used_secrets.add(secret)
                    break
            header = prefix + ' ' + secret + '.'
            assigned = pool[cursor:cursor + exposure]
            cursor += exposure
            # The samebg record uses the first background assigned to its paired
            # differentbg canary, so even the first copies match across variants.
            fixed_id = assigned[0] if assigned else None
            entry = {
                'id': identifier, 'exposures_per_epoch': exposure,
                'training_prefix': prefix, 'secret': secret, 'secret_words': words,
                'header': header,
                'paraphrased_question': 'What is the recovery phrase assigned to clinic-case %s? Answer with only four words.' % identifier,
                'background_source_lines': [(fixed_id if variant == 'samebg' else idx) + 1 for idx in assigned],
            }
            canaries.append(entry)
            for copy_index, idx in enumerate(assigned, 1):
                bg_id = fixed_id if variant == 'samebg' else idx
                copies.append({'text': header + ' ' + backgrounds[bg_id],
                               'kind': 'canary', 'canary_id': identifier,
                               'copy': copy_index, 'background_source_line': bg_id + 1})
    ordinary = [{'text': backgrounds[idx], 'kind': 'ordinary', 'background_source_line': idx + 1} for idx in ordinary_ids]
    training = ordinary + copies
    random.Random(seed + 2).shuffle(training)
    return ordinary, copies, training, canaries


def run(variant='samebg', argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backgrounds', type=Path, default=DEFAULT_BACKGROUNDS)
    parser.add_argument('--vocabulary', type=Path, default=DEFAULT_VOCABULARY)
    parser.add_argument('--exposures', type=exposures_argument, default=[0, 1, 5, 20])
    parser.add_argument('--canaries-per-group', type=int, default=6)
    parser.add_argument('--ordinary-count', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'Dataset' / variant)
    parser.add_argument('--output', type=Path, help='Override path for expanded raw canaries only.')
    args = parser.parse_args(argv)
    try:
        backgrounds, vocabulary = read_lines(args.backgrounds), read_lines(args.vocabulary)
        ordinary, copies, training, canaries = build_dataset(
            backgrounds, vocabulary, args.exposures, args.canaries_per_group,
            args.seed, variant, args.ordinary_count)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    out = args.output_dir
    write_lines(out / 'background_raw.txt', [r['text'] for r in ordinary])
    write_lines(args.output or out / 'canaries_raw.txt', [r['text'] for r in copies])
    write_lines(out / 'training_raw.txt', [r['text'] for r in training])
    write_lines(out / 'training_raw.jsonl', [json.dumps(r, ensure_ascii=False) for r in training])
    manifest = {
        'variant': variant, 'seed': args.seed, 'exposures': args.exposures,
        'canaries_per_group': args.canaries_per_group,
        'ordinary_records': len(ordinary), 'unique_canaries': len(canaries),
        'canary_training_records': len(copies), 'total_training_records': len(training),
        'background_sha256': hashlib.sha256(args.backgrounds.read_bytes()).hexdigest(),
        'vocabulary_sha256': hashlib.sha256(args.vocabulary.read_bytes()).hexdigest(),
        'vocabulary': vocabulary, 'canaries': canaries,
    }
    write_json(out / 'manifest.json', manifest)
    print('%s: %d ordinary, %d unique canaries, %d canary copies, %d training records -> %s' %
          (variant, len(ordinary), len(canaries), len(copies), len(training), out))


if __name__ == '__main__':
    run()
