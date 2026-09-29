import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from autoschema_experiment.common import digest, load_articles
from autoschema_experiment.event_extraction.extract import validate, RELATIONS, span
from autoschema_experiment.concept_induction.induce import validate as validate_concepts, objects_from_events
from autoschema_experiment.llm import Model
from autoschema_experiment.run import main


class Contracts(unittest.TestCase):
    def setUp(self):
        text = 'In mice, MASH activates HSCs, causing fibrosis.'
        self.article = {'pmid': '123', 'source_text': text, 'source_hash': digest(text)}
        self.event = {'local_id': 'e1', 'sentence': 'MASH activates HSCs in mice.',
                      'participants': [{'mention': 'MASH'}, {'mention': 'HSCs'}],
                      'evidence': text, 'assertion': 'asserted', 'context': 'mice'}
        second = {**self.event, 'local_id': 'e2', 'sentence': 'Fibrosis develops in mice.'}
        self.payload = {'events': [self.event, second], 'event_relations': [
            {'head': 'e1', 'tail': 'e2', 'relation': 'AS_A_RESULT', 'evidence': text}]}

    def test_exact_provenance_and_stable_ids(self):
        es, rs, bad = validate(self.article, self.payload, {})
        self.assertEqual((len(es), len(rs), len(bad)), (2, 1, 0))
        self.assertEqual(validate(self.article, self.payload, {})[0], es)
        self.assertEqual(rs[0]['head_event_id'], es[0]['event_id'])
        for e in es:
            q = e['evidence']
            self.assertEqual(e['source_text'][q['char_start']:q['char_end']], q['text'])
            for p in e['participants']:
                self.assertEqual(e['source_text'][p['char_start']:p['char_end']], p['mention'])

    def test_fabricated_evidence_and_dangling_endpoint_rejected(self):
        self.payload['events'][0]['evidence'] = 'Invented assertion.'
        es, rs, bad = validate(self.article, self.payload, {})
        self.assertEqual((len(es), len(rs), len(bad)), (1, 0, 2))

    def test_duplicate_ids_reject_all_ambiguous_events(self):
        self.payload['events'][1]['local_id'] = 'e1'
        es, rs, bad = validate(self.article, self.payload, {})
        self.assertEqual((len(es), len(rs), len(bad)), (0, 0, 3))

    def test_labels_and_self_links(self):
        for label in RELATIONS:
            self.payload['event_relations'][0]['relation'] = label
            self.assertEqual(len(validate(self.article, self.payload, {})[1]), 1)
        self.payload['event_relations'][0]['relation'] = 'CAUSES'
        self.assertEqual(len(validate(self.article, self.payload, {})[1]), 0)
        self.payload['event_relations'][0].update(relation='BEFORE', tail='e1')
        self.assertEqual(len(validate(self.article, self.payload, {})[1]), 0)

    def test_participant_must_be_word_in_evidence(self):
        for mention in ['unmentioned gene', 'ASH']:
            self.payload['events'][0]['participants'] = [{'mention': mention}]
            self.assertEqual(len(validate(self.article, self.payload, {})[0]), 1)

    def test_normalized_quotes_not_silently_accepted(self):
        with self.assertRaises(ValueError):
            span('MASH activates HSCs.', 'MASH activates  HSCs.')
        # Unicode normalization can change string length; raw offsets must still match.
        with self.assertRaises(ValueError):
            span('fiber', 'ﬁber')

    def test_malformed_shapes(self):
        with self.assertRaises(ValueError):
            validate(self.article, {'events': {}, 'event_relations': []}, {})
        self.payload['events'].append(None)
        self.payload['event_relations'].append({'head': [], 'tail': 'e1', 'relation': 'BEFORE'})
        self.assertEqual(len(validate(self.article, self.payload, {})[2]), 2)

    def test_negation_is_preserved(self):
        self.event['assertion'] = 'negated'
        self.event['sentence'] = 'MASH did not activate HSCs in mice.'
        self.assertEqual(validate(self.article, self.payload, {})[0][0]['assertion'], 'negated')
        # Structural validation is not a semantic entailment judge.
        self.assertEqual(validate(self.article, self.payload, {})[0][0]['validation_status'], 'source_grounded_pending_review')

    def test_three_concept_types_no_merging(self):
        es, rs, _ = validate(self.article, self.payload, {})
        objects = objects_from_events(es, rs)
        self.assertEqual({o['source_type'] for o in objects}, {'Entity', 'Event', 'Relation'})
        payload = {'concepts': [{'source_id': o['source_id'], 'concepts': ['Biological process', 'Pathology']} for o in objects]}
        cs, bad = validate_concepts(objects, payload, {})
        self.assertEqual(len(cs), len(objects)*2)
        self.assertFalse(bad)
        self.assertEqual(len({c['concept_id'] for c in cs}), len(cs))

    def test_concept_coverage_and_unknown_ids(self):
        obj = {'source_id': 'e1', 'source_type': 'Entity', 'source_pmid': '123', 'source_label': 'HSC', 'context': 'HSC activation'}
        cs, bad = validate_concepts([obj], {'concepts': [{'source_id': 'x', 'concepts': ['A', 'B']}]}, {})
        self.assertFalse(cs)
        self.assertEqual(len(bad), 2)

    def test_input_duplicate_pmid(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'in.jsonl'
            p.write_text((json.dumps({'pmid': '123', 'abstract': 'text'})+'\n')*2)
            with self.assertRaises(ValueError):
                load_articles(p)

    def test_explicit_env_file_overrides_stale_inherited_key(self):
        import os
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'selected.env'
            path.write_text('GEMINI_API_KEY=' + 'new-key-' + 'x'*24 + '\n'
                'GEMINI_API_BASE=https://example.test/v1\nGEMINI_MODEL=test-model\n')
            with patch.dict(os.environ, {'GEMINI_API_KEY': 'stale-key'}, clear=False):
                model = Model(Path(d)/'out', env_file=path, model_role='extraction')
                self.assertEqual(model.registry.specs['primary'].api_key, 'new-key-' + 'x'*24)
                self.assertEqual(model.registry.specs['primary'].api_base, 'https://example.test/v1')
                self.assertEqual(model.model, 'test-model')

    def test_replay_no_network_and_hash_binding(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); raw=root/'raw'; raw.mkdir(); output=root/'out'; output.mkdir()
            record={'request_hash': digest({'system': 'system', 'payload': {'a': 1}}), 'payload': {'ok': True}, 'metadata': {'model': 'fixture'}}
            (raw/'events_123.json').write_text(json.dumps(record))
            with patch('socket.socket', side_effect=AssertionError('network forbidden')):
                model=Model(output, replay=raw)
                self.assertTrue(model.call('events','123','system',{'a':1})[0]['ok'])
                with self.assertRaises(ValueError):
                    model.call('events','123','changed',{'a':1})

    def test_end_to_end_replay_writes_all_artifacts_without_network(self):
        from autoschema_experiment.event_extraction.extract import PROMPT as EPROMPT
        from autoschema_experiment.concept_induction.induce import PROMPT as CPROMPT
        from cognitive_agent.evidence_units import ArticleEvidenceReader
        from cognitive_agent.abbreviation_detector import AbbreviationDetector
        from autoschema_experiment.concept_induction.induce import objects_from_events
        text = self.article['source_text']
        article = {'pmid': '123', 'title': 'Mouse study', 'abstract': text}
        event_payload = copy.deepcopy(self.payload)
        events, relations, rejected = validate(self.article, event_payload, {})
        self.assertFalse(rejected)
        objects = objects_from_events(events, relations)
        concepts_payload = {'concepts': [
            {'source_id': o['source_id'], 'concepts': ['Biological process', 'Experimental finding']}
            for o in objects]}
        event_request = {'pmid': '123', 'title': 'Mouse study', 'source_text': text,
            'evidence_units': [u.to_dict() for u in ArticleEvidenceReader().read(text)],
            'abbreviations': AbbreviationDetector().detect(text).to_dict()}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); inp=root/'input.jsonl'; inp.write_text(json.dumps(article)+'\n')
            replay=root/'replay'; replay.mkdir()
            for stage, prompt, request, response in [
                ('events', EPROMPT, event_request, event_payload),
                ('concepts', CPROMPT, {'objects': objects}, concepts_payload)]:
                (replay/f'{stage}_123.json').write_text(json.dumps({
                    'request_hash': digest({'system': prompt, 'payload': request}),
                    'payload': response, 'metadata': {'model': 'test-fixture'}}))
            out=root/'out'
            with patch('socket.socket', side_effect=AssertionError('network forbidden')):
                self.assertEqual(main(['--input',str(inp),'--replay',str(replay),'--output',str(out)]), 0)
            self.assertEqual(len(json.loads((out/'events.jsonl').read_text().splitlines()[0])['participants']), 2)
            self.assertEqual(len((out/'event_relations.jsonl').read_text().splitlines()), 1)
            self.assertEqual(len((out/'concepts.jsonl').read_text().splitlines()), len(objects)*2)
            self.assertIn('pending', (out/'manual_review.csv').read_text(encoding='utf-8-sig'))
            self.assertEqual(json.loads((out/'manifest.json').read_text())['mode'], 'replay')

    def test_existing_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):
                main(['--output',d])


if __name__ == '__main__':
    unittest.main()
