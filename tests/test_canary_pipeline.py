"""Dependency-free invariant tests; real tokenizer checked separately by the CLI."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from create_canaries_samebg import build_dataset
from normalize_records import normalize_record


class CharacterTokenizer:
    """Deterministic test double for truncation and protected-header boundaries."""
    def encode(self, text, **kwargs):
        return list(map(ord, text))

    def decode(self, ids, **kwargs):
        return ''.join(map(chr, ids))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.backgrounds = ['Distinct background %d with enough explanatory text.' % n for n in range(256)]
        self.words = ['fruit' + chr(97 + n // 26) + chr(97 + n % 26) for n in range(32)]

    def build(self, variant):
        return build_dataset(self.backgrounds, self.words, [0, 1, 5, 20], variant=variant)

    def test_counts_controls_and_unique_secrets(self):
        ordinary, copies, training, manifest = self.build('samebg')
        self.assertEqual([len(ordinary), len(copies), len(training), len(manifest)], [100, 156, 256, 24])
        self.assertEqual(len({c['secret'] for c in manifest}), 24)
        for canary in manifest:
            rows = [r for r in copies if r['canary_id'] == canary['id']]
            self.assertEqual(len(rows), canary['exposures_per_epoch'])
        self.assertEqual(sum(c['exposures_per_epoch'] == 0 for c in manifest), 6)

    def test_matched_variants_and_disjoint_backgrounds(self):
        same, different = self.build('samebg'), self.build('differentbg')
        self.assertEqual(same[0], different[0])
        self.assertEqual(same, self.build('samebg'))
        ordinary_ids = {r['background_source_line'] for r in same[0]}
        self.assertFalse(ordinary_ids & {r['background_source_line'] for r in different[1]})
        self.assertEqual(len({r['background_source_line'] for r in different[1]}), 156)
        for a, b in zip(same[3], different[3]):
            self.assertEqual((a['id'], a['secret']), (b['id'], b['secret']))
            sa = [r['text'] for r in same[1] if r['canary_id'] == a['id']]
            db = [r['text'] for r in different[1] if r['canary_id'] == b['id']]
            n = a['exposures_per_epoch']
            self.assertEqual(len(set(sa)), int(n > 0))
            self.assertEqual(len(set(db)), n)
            if n:
                self.assertEqual(sa[0], db[0])
        self.assertEqual([(r['kind'], r.get('canary_id'), r.get('copy')) for r in same[2]],
                         [(r['kind'], r.get('canary_id'), r.get('copy')) for r in different[2]])

    def test_invalid_inputs(self):
        for backgrounds, words, levels in [
            (self.backgrounds[:-1], self.words, [0, 1, 5, 20]),
            (self.backgrounds, self.words[:-1], [0, 1, 5, 20]),
            (self.backgrounds, self.words, [-1]),
            (self.backgrounds, self.words, [1, 1]),
            (self.backgrounds + self.backgrounds[:1], self.words, [0, 1]),
        ]:
            with self.assertRaises(ValueError):
                build_dataset(backgrounds, words, levels)

    def test_exact_length_and_secret_preservation(self):
        header = 'The private recovery phrase for clinic-case patient-001 is apple pear plum kiwi.'
        source = header + ' ' + 'heart information ' * 20
        text, ids = normalize_record(source, CharacterTokenizer())
        self.assertEqual(len(ids), 96)
        self.assertTrue(text.startswith(header + ' '))
        self.assertTrue(source.startswith(text))
        self.assertEqual(CharacterTokenizer().encode(text), ids)
        with self.assertRaises(ValueError):
            normalize_record(source, CharacterTokenizer(), len(header))

    def test_short_malformed_and_unstable_records(self):
        with self.assertRaises(ValueError):
            normalize_record('short', CharacterTokenizer())
        with self.assertRaises(ValueError):
            normalize_record('The private recovery phrase has no valid secret.' * 5, CharacterTokenizer())
        class UnstableTokenizer(CharacterTokenizer):
            def decode(self, ids, **kwargs):
                return super().decode(ids, **kwargs) + 'x'
        with self.assertRaises(ValueError):
            normalize_record('background ' * 100, UnstableTokenizer())


if __name__ == '__main__':
    unittest.main()
