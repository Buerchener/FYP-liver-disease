"""Durable REFINE state machine: settle the current sibling set BEFORE descent."""
from copy import deepcopy
from pathlib import Path
from ..models import to_dict
from . import contracts
from .context import concept_catalog
from .contracts import BudgetExceeded, CandidateError, ContextLimit, ContractError, ProviderError
from .graph import (accept_candidate, build_candidate, children, load_taxonomy,
                    local_signature, shortest_depths, structure_signature)
from .storage import atomic_json, read_checkpoint, save_checkpoint


def frame(parent, depth):
    return {'parent_id': parent, 'depth': depth, 'phase': 'enter', 'round': 0,
            'attempt': 1, 'diagnosed_signatures': [], 'attempted_candidate_signatures': [], 'feedback': []}


def initial_state(taxonomy, run_fingerprint):
    return {'format': 'recursive-induction/1', 'run_fingerprint': run_fingerprint,
            'taxonomy': to_dict(taxonomy), 'stack': [], 'pending_roots': list(taxonomy.predefined_roots),
            'completed': {}, 'context_visits': {}, 'accepted_signatures': [structure_signature(taxonomy)],
            'initial_induced_nodes': sum(n.origin_kind == 'induced_intermediate' for n in taxonomy.nodes),
            'sequence': 0, 'status': 'running', 'accepted_candidates': 0, 'rejected_candidates': 0}


class RecursiveRefiner:
    def __init__(self, directory, state, journal, context_builder, clusterer, config):
        self.directory, self.state = Path(directory), state
        self.journal, self.context_builder, self.clusterer, self.config = journal, context_builder, clusterer, config

    def _cursor(self, current):
        return {k: current[k] for k in ('parent_id', 'depth', 'round', 'attempt')}

    def _call(self, stage, payload):
        self.context_builder.check(payload)
        return self.journal.call(stage, payload)

    def _stop(self, current, taxonomy, reason, stable=False):
        current['phase'] = 'descend'
        current['stop_signature'] = local_signature(taxonomy, current['parent_id'])
        current['descendants'] = children(taxonomy, current['parent_id'])
        current['child_index'] = 0
        for key in ('candidate', 'changes', 'diagnosis', 'groups', 'context'):
            current.pop(key, None)
        return {'action': 'stop_level', 'reason': reason, 'semantically_stable': stable,
                'continue_into_children': True}

    def _reject(self, current, taxonomy, reason, response=None):
        self.state['rejected_candidates'] += 1
        feedback = {'attempt': current['attempt'], 'reason': reason,
                    'proposal': current.get('proposal'), 'validation': response}
        current['feedback'].append(feedback)
        candidate = current.pop('candidate', None)
        changes = current.pop('changes', None)
        current['attempt'] += 1
        current['phase'] = 'propose'
        result = {'action': 'reject_candidate', 'reason': reason, 'feedback': feedback,
                  'candidate_snapshot': candidate, 'changes': changes, 'original_structure_retained': True,
                  'means_optimal': False}
        if current['attempt'] > self.config['limits']['candidate_attempts']:
            result['level_stop'] = self._stop(current, taxonomy, 'candidate_attempts_exhausted')
        return result

    def _already_done(self, node, signature, depth):
        prior_depth = self.state['completed'].get(node, {}).get(signature)
        return prior_depth is not None and prior_depth <= depth

    def _finish(self, current):
        history = self.state['completed'].setdefault(current['parent_id'], {})
        signature = current['stop_signature']
        history[signature] = min(history.get(signature, current['depth']), current['depth'])
        self.state['stack'].pop()

    def _transition(self, taxonomy):
        state, limits = self.state, self.config['limits']
        if not state['stack']:
            if state['pending_roots']:
                parent = state['pending_roots'].pop(0)
                state['stack'].append(frame(parent, 0))
                return {'action': 'schedule_root', 'parent_id': parent}
            # A shared node's ancestor context can change while another root is
            # refined. Revisit stale contexts, not every occurrence of a DAG node.
            depths = shortest_depths(taxonomy)
            for parent, depth in sorted(depths.items(), key=lambda pair: (pair[1], pair[0])):
                signature = local_signature(taxonomy, parent)
                if depth <= limits['max_depth'] and not self._already_done(parent, signature, depth):
                    state['stack'].append(frame(parent, depth))
                    return {'action': 'schedule_stale_dag_context', 'parent_id': parent, 'depth': depth}
            state['status'] = 'completed'
            return {'action': 'complete_run'}
        current = state['stack'][-1]
        parent, phase = current['parent_id'], current['phase']
        detail = {'parent_id': parent, 'phase': phase, 'cursor': self._cursor(current)}
        signature = local_signature(taxonomy, parent)
        if phase == 'enter':
            if current['depth'] > limits['max_depth']:
                state['stack'].pop()
                return dict(detail, action='skip_depth_limit', reason='maximum_recursion_depth')
            if self._already_done(parent, signature, current['depth']):
                state['stack'].pop()
                return dict(detail, action='skip_unchanged_shared_context')
            visits = state['context_visits'].get(parent, 0)
            state['context_visits'][parent] = visits + 1
            if visits >= limits['context_visits_per_node']:
                return dict(detail, **self._stop(current, taxonomy, 'maximum_context_visits'))
            if any(f['parent_id'] == parent for f in state['stack'][:-1]):
                raise ValueError('Active recursion cycle')
            current['phase'] = 'diagnose'
            return dict(detail, action='enter_parent')
        if phase == 'diagnose':
            direct = children(taxonomy, parent)
            if len(direct) < 2:
                return dict(detail, **self._stop(current, taxonomy, 'fewer_than_two_children', True))
            if sum(n.origin_kind == 'induced_intermediate' for n in taxonomy.nodes) - state['initial_induced_nodes'] >= limits['new_nodes']:
                return dict(detail, **self._stop(current, taxonomy, 'maximum_new_nodes'))
            if current['round'] >= limits['iterations_per_parent']:
                return dict(detail, **self._stop(current, taxonomy, 'maximum_parent_iterations'))
            if signature in current['diagnosed_signatures']:
                return dict(detail, **self._stop(current, taxonomy, 'repeated_local_structure'))
            current['round'] += 1; current['attempt'] = 1
            current['feedback'] = []
            current['diagnosed_signatures'].append(signature)
            try:
                context = self.context_builder.build(taxonomy, parent)
                response = contracts.diagnosis(self._call('diagnosis', {'cursor': self._cursor(current), 'context': context}), set(direct))
            except (ContractError, BudgetExceeded, ContextLimit) as error:
                return dict(detail, **self._stop(current, taxonomy, type(error).__name__ + ':' + str(error)))
            detail.update(diagnosis=response, llm_ref=self.journal.last_ref)
            if response['confidence'] < self.config['thresholds']['diagnosis_confidence']:
                return dict(detail, **self._stop(current, taxonomy, 'low_diagnosis_confidence'))
            if not response['need_induction']:
                return dict(detail, **self._stop(current, taxonomy, 'llm_no_induction', True))
            try:
                groups, metadata = self.clusterer.cluster(context, response, taxonomy)
                for group in groups:
                    contracts.ids(group['child_ids'], set(direct))
                if not groups:
                    return dict(detail, **self._stop(current, taxonomy, 'no_candidate_groups'))
            except (ValueError, KeyError, TypeError) as error:
                return dict(detail, **self._stop(current, taxonomy, 'invalid_clusters:' + str(error)))
            current.update(context=context, diagnosis=response, groups=groups, phase='propose')
            return dict(detail, action='diagnose_and_group', groups=groups, clustering=metadata)
        if phase == 'propose':
            current.pop('proposal', None)
            try:
                response = contracts.proposal(self._call('induction', {
                    'cursor': self._cursor(current), 'context': current['context'],
                    'diagnosis': current['diagnosis'], 'groups': current['groups'],
                    'existing_concept_catalog': concept_catalog(taxonomy), 'rejected_candidates': current['feedback']}), current['groups'])
                current['proposal'] = response
                if response['confidence'] < self.config['thresholds']['induction_confidence']:
                    raise CandidateError('low_induction_confidence')
                candidate, changes = build_candidate(taxonomy, parent, response, limits['new_nodes'] + state['initial_induced_nodes'])
                if structure_signature(candidate) in state['accepted_signatures']:
                    raise CandidateError('repeated_global_structure')
                if structure_signature(candidate) in current['attempted_candidate_signatures']:
                    raise CandidateError('repeated_rejected_candidate')
                current['attempted_candidate_signatures'].append(structure_signature(candidate))
            except (BudgetExceeded, ContextLimit) as error:
                return dict(detail, **self._stop(current, taxonomy, type(error).__name__ + ':' + str(error)))
            except (ContractError, CandidateError, ValueError) as error:
                return dict(detail, **self._reject(current, taxonomy, str(error)))
            current.update(candidate=to_dict(candidate), changes=changes, phase='validate')
            return dict(detail, action='candidate_constructed', changes=changes,
                        candidate_snapshot=to_dict(candidate), llm_ref=self.journal.last_ref)
        if phase == 'validate':
            candidate = load_taxonomy(current['candidate'])
            try:
                response = contracts.validation(self._call('validation', {
                    'cursor': self._cursor(current), 'before_context': self.context_builder.build(taxonomy, parent),
                    'after_context': self.context_builder.build(candidate, parent),
                    'proposed_changes': current['changes'], 'existing_concept_catalog': concept_catalog(taxonomy)}),
                    current['changes']['added_edges'], set(children(candidate, parent)))
                if response['decision'] == 'REJECT':
                    return dict(detail, llm_ref=self.journal.last_ref,
                                **self._reject(current, taxonomy, 'llm_rejected_candidate', response))
                if response['confidence'] < self.config['thresholds']['validation_confidence']:
                    raise CandidateError('low_validation_confidence')
                accepted = accept_candidate(taxonomy, candidate, current['changes'], response, self.journal.last_ref)
            except (BudgetExceeded, ContextLimit) as error:
                saved = current['candidate']
                return dict(detail, candidate_snapshot=saved, **self._stop(current, taxonomy, type(error).__name__ + ':' + str(error)))
            except (ContractError, CandidateError, ValueError) as error:
                return dict(detail, **self._reject(current, taxonomy, str(error)))
            changes = current['changes']
            state['taxonomy'] = to_dict(accepted)
            state['accepted_signatures'].append(structure_signature(accepted))
            state['accepted_candidates'] += 1
            for key in ('candidate', 'changes', 'proposal', 'context', 'diagnosis', 'groups'):
                current.pop(key, None)
            detail.update(action='accept_candidate', decision=response['decision'], validation=response,
                          changes=changes, candidate_snapshot=to_dict(candidate), llm_ref=self.journal.last_ref)
            if response['decision'] == 'ACCEPT_AND_CONTINUE':
                # The essential ordering: no child is scheduled here.
                current['phase'] = 'diagnose'
            else:
                detail['level_stop'] = self._stop(current, accepted, 'llm_accept_and_stop', True)
            return detail
        if phase == 'descend':
            if current['child_index'] < len(current['descendants']):
                child = current['descendants'][current['child_index']]
                current['child_index'] += 1
                state['stack'].append(frame(child, current['depth'] + 1))
                return dict(detail, action='descend_to_child', child_id=child)
            self._finish(current)
            return dict(detail, action='finish_parent')
        raise ValueError('Unknown checkpoint phase: ' + phase)

    def step(self):
        """Commit exactly one transition. Provider failures leave the checkpoint intact."""
        old_state = deepcopy(self.state)
        before = deepcopy(self.state['taxonomy'])
        try:
            detail = self._transition(load_taxonomy(before))
            sequence = self.state['sequence'] + 1
            step_dir = self.directory / 'steps' / ('%06d' % sequence)
            candidate = detail.pop('candidate_snapshot', None)
            atomic_json(step_dir / 'before.json', before)
            atomic_json(step_dir / 'after.json', self.state['taxonomy'])
            if candidate is not None:
                atomic_json(step_dir / 'candidate.json', candidate)
                detail['candidate_snapshot_ref'] = 'candidate.json'
            detail.update(sequence=sequence, before_ref='before.json', after_ref='after.json')
            atomic_json(step_dir / 'change.json', detail)
            self.state['sequence'] = sequence
            save_checkpoint(self.directory / 'checkpoint.json', self.state)
            return detail
        except BaseException:
            self.state = old_state
            raise

    def run(self, step_limit=None, stop_requested=None):
        count = 0
        while self.state['status'] != 'completed' and (step_limit is None or count < step_limit):
            if stop_requested is not None and stop_requested():
                break
            self.step(); count += 1
        return self.state
