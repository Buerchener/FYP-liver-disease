"""Independent experiment lifecycle. No imports from extraction or model credentials."""
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
from .clustering import make_clusterer
from .config import read_config, validate_config
from .context import ContextBuilder
from .contracts import ProviderError
from .dataset import load_dataset, validate_dataset
from .graph import assert_preserved, digest, load_taxonomy
from .journal import LLMJournal
from .providers import MockProvider, OpenAICompatibleProvider
from .recursive import RecursiveRefiner, initial_state
from .storage import atomic_json, read_checkpoint, read_json, run_lock, save_checkpoint
from .visualize import render_run
from ..taxonomy.dag import validate_dag

PACKAGE = Path(__file__).resolve().parents[1]


def code_files():
    return sorted(p for p in PACKAGE.rglob('*.py') if not any(part in ('runs', '__pycache__') for part in p.relative_to(PACKAGE).parts))


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _check_live(config, live):
    if config['provider']['kind'] == 'openai_compatible' and not live:
        raise ValueError('Live model configuration requires explicit --live')


def create_run(config_path, output_parent=None, run_id=None, live=False, step_limit=None):
    config = read_config(config_path)
    _check_live(config, live)
    dataset = load_dataset(config['dataset'])
    taxonomy = validate_dataset(dataset, config['require_semantic_input_frozen'])
    if config['provider']['kind'] == 'mock' and dataset['data_origin'] != 'synthetic':
        raise ValueError('The supplied rule mock is restricted to Synthetic Data')
    prompts = {stage: Path(path).read_text(encoding='utf-8') for stage, path in config['prompts'].items()}
    mock_plan = read_json(config['provider']['plan_file']) if config['provider']['kind'] == 'mock' else None
    vectors = read_json(config['clustering']['vectors_file']) if config['clustering'].get('vectors_file') else {}
    make_clusterer(config['clustering'], vectors)
    name = run_id or datetime.now(timezone.utc).strftime('task2-%Y%m%dT%H%M%S%fZ')
    if Path(name).name != name or name in ('.', '..'):
        raise ValueError('run_id must be one directory name')
    directory = Path(output_parent or config['output_root']).resolve() / name
    directory.mkdir(parents=True, exist_ok=False)
    with run_lock(directory):
        for name, value in [('config.json',config), ('input_dataset.json',dataset), ('prompts.json',prompts),
                            ('mock_plan.json',mock_plan), ('embeddings.json',vectors)]:
            atomic_json(directory / 'frozen' / name, value)
        hashes = {str(p.relative_to(directory)): file_hash(p) for p in (directory/'frozen').iterdir()}
        codes = {}
        for path in code_files():
            relative = path.relative_to(PACKAGE)
            target = directory / 'code_snapshot' / 'event_taxonomy' / relative
            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(path.read_bytes())
            codes[str(relative)] = file_hash(path)
            hashes[str(target.relative_to(directory))] = file_hash(target)
        manifest = {'schema_version':'event-taxonomy-induction-run/1', 'files_sha256':hashes,
                    'code_sha256':codes, 'data_origin':dataset['data_origin'],
                    'provider_kind':config['provider']['kind'],
                    'semantic_accuracy_evaluated':False, 'source_config_path':str(Path(config_path).resolve())}
        fingerprint = digest(manifest)
        atomic_json(directory/'manifest.json', {'manifest':manifest,'sha256':fingerprint})
        atomic_json(directory/'initial_taxonomy.json',dataset['taxonomy'])
        save_checkpoint(directory/'checkpoint.json',initial_state(taxonomy,fingerprint))
        _execute(directory, config, dataset, prompts, mock_plan, vectors, step_limit)
    return directory


def verify_run(directory):
    envelope = read_json(directory / 'manifest.json'); manifest = envelope['manifest']
    if digest(manifest) != envelope['sha256']:
        raise ValueError('Manifest checksum mismatch')
    for name, checksum in manifest['files_sha256'].items():
        if file_hash(directory / name) != checksum:
            raise ValueError('Frozen artifact changed: ' + name)
    actual = {str(p.relative_to(PACKAGE)):file_hash(p) for p in code_files()}
    if actual != manifest['code_sha256']:
        raise ValueError('Algorithm code changed. Resume with the saved code_snapshot on PYTHONPATH.')
    state = read_checkpoint(directory/'checkpoint.json')
    if state['run_fingerprint'] != envelope['sha256']:
        raise ValueError('Checkpoint belongs to another experiment')
    frozen_input = read_json(directory/'frozen/input_dataset.json')['taxonomy']
    if read_json(directory/'initial_taxonomy.json') != frozen_input:
        raise ValueError('Initial taxonomy artifact changed')
    assert_preserved(load_taxonomy(frozen_input),load_taxonomy(state['taxonomy']),accepted=True)
    return state


def resume_run(directory, live=False, step_limit=None):
    directory = Path(directory).resolve()
    with run_lock(directory):
        verify_run(directory)
        config = validate_config(read_json(directory/'frozen/config.json'))
        _check_live(config, live)
        _execute(directory,config,read_json(directory/'frozen/input_dataset.json'),read_json(directory/'frozen/prompts.json'),
                 read_json(directory/'frozen/mock_plan.json'),read_json(directory/'frozen/embeddings.json'),step_limit)
    return directory


def _execute(directory, config, dataset, prompts, mock_plan, vectors, step_limit):
    if step_limit is not None and (type(step_limit) is not int or step_limit < 1):
        raise ValueError('step_limit must be positive')
    provider = MockProvider(mock_plan) if config['provider']['kind'] == 'mock' else OpenAICompatibleProvider(config['provider'])
    journal = LLMJournal(directory/'llm',provider,prompts,config['stages'],config['limits']['llm_calls'])
    engine = RecursiveRefiner(directory, read_checkpoint(directory/'checkpoint.json'), journal,
                             ContextBuilder(dataset,config['context']),make_clusterer(config['clustering'],vectors),config)
    try:
        pause_file = os.environ.get('EVENT_TAXONOMY_PAUSE_FILE')
        state = engine.run(step_limit, stop_requested=lambda: bool(pause_file and Path(pause_file).exists()))
    except (ProviderError, KeyboardInterrupt) as error:
        atomic_json(directory/'execution_status.json', {'status':'interrupted','reason':str(error),
                    'resume_command':'python -m event_taxonomy resume --run ' + str(directory) + (' --live' if config['provider']['kind'] != 'mock' else '')})
        raise
    before, after = load_taxonomy(dataset['taxonomy']), load_taxonomy(state['taxonomy'])
    assert_preserved(before,after,accepted=True)
    # Validate provenance even for newly induced concepts, not only the old subset.
    output_dataset = dict(dataset, taxonomy=state['taxonomy'])
    validate_dataset(output_dataset, config['require_semantic_input_frozen'])
    report = dict(validate_dag(after),data_origin=dataset['data_origin'],provider_kind=config['provider']['kind'],
        execution_status=state['status'],initial_nodes=len(before.nodes),final_nodes=len(after.nodes),
        initial_edges=len(before.parent_child_edges),final_edges=len(after.parent_child_edges),
        original_concepts_preserved=len(before.frozen_input_concept_ids),original_concept_retention=1.0,
        original_events_preserved=len(dataset['events']),original_evidence_preserved=len(dataset['evidence']),
        instance_links_preserved=True, original_concept_evidence_mapping_preserved=True,
        accepted_candidates=state['accepted_candidates'],rejected_candidates=state['rejected_candidates'],
        llm_request_attempts=len(list((directory/'llm').glob('*.json'))),
        committed_steps=state['sequence'],semantic_accuracy_evaluated=False,
        note='Synthetic mock demonstrates control flow only.' if dataset['data_origin']=='synthetic' else 'LLM-validated structure still requires independent evaluation.')
    atomic_json(directory/'final_taxonomy.json',state['taxonomy'])
    atomic_json(directory/'preservation_report.json',report)
    changes = [read_json(directory/'steps'/('%06d'%i)/'change.json') for i in range(1,state['sequence']+1)]
    atomic_json(directory/'change_log.json',changes)
    atomic_json(directory/'execution_status.json',{'status':state['status'] if state['status']=='completed' else 'paused',
                                                 'committed_steps':state['sequence']})
    render_run(directory,state,report)
