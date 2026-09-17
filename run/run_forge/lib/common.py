"""Bundle identity, runtime environment, and immutable measurement inputs."""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parents[1]
UPSTREAM_SHA = 'cd9c5850699b0550c2aa06be83c3645cf4e98e24'
API_HASHES = {
    'src/kernel_agents/cli.py': '11150faf70e9fa24bc489094baef2ee2fb021dbc9437f587fd4aae00b047f9fd',
    'src/kernel_agents/loop/task_preparer.py': '8f7735fdfc6874c098200cce697479a49cac5625a8752ffc6960d949113a771a',
    'src/kernel_agents/loop/program.py': 'b153c6718910a1045bafc68701165727b4a6ff235a35790b28d3b5daa7854b6e',
}
RECIPES = {
    'mxfp4_moe_experts_001': dict(template='moe', parent='mxfp4_moe_chain_001',
        kernel='measurement/kernel.py', nproc=1, fellow='flydsl-fellow', framework='aiter',
        operator='mxfp4_moe_expert_chain', snapshot_aiter=True,
        targets=['moe_forward', 'fused_moe', 'flydsl_moe1', 'flydsl_moe2'],
        sources=['measurement/kernel.py', 'measurement/candidate_config.csv', 'aiter/aiter/fused_moe.py',
                 'aiter/aiter/ops/flydsl/moe_kernels.py', 'aiter/aiter/ops/flydsl/kernels/moe_sorting_kernel.py'],
        editable=['measurement/kernel.py', 'measurement/candidate_config.csv',
                  'aiter/aiter/fused_moe.py', 'aiter/aiter/ops/flydsl/'],
        scope='Captured TP0 MXFP4 expert chain: sort, activation quantization, GEMM1/GEMM2 and route accumulation. '
              'M=192 H=6144 I=512 E=257 topk=9. Three actual layer snapshots (3/39/77). '
              'Router projection/top-k selection, collectives and e2e serving are outside scope.',
        workload_counts={},
        errata=['Source handoff readiness resolved by graph-replay capture of real inputs and routing.',
                'The benchmark uses three layer snapshots with equal case weight; it does not claim all-layer/e2e coverage.',
                '192 is the input tensor row count; 256 is only the observed tuning bucket.']),
    'tp4_bf16_allreduce_001': dict(template='allreduce', parent='tp4_bf16_allreduce_001',
        kernel='aiter/csrc/include/custom_all_reduce.cuh', nproc=4, fellow='hip-fellow',
        framework='aiter', operator='custom_all_reduce', snapshot_aiter=True,
        targets=['cross_device_reduce_2stage'],
        sources=['aiter/csrc/include/custom_all_reduce.cuh', 'aiter/csrc/kernels/custom_all_reduce.cu',
                 'aiter/aiter/dist/device_communicators/custom_all_reduce.py'],
        editable=['aiter/csrc/', 'aiter/aiter/dist/'],
        scope='Raw TP4 out-of-place BF16 all-reduce, three callsite input classes; excludes SGLang writeback and e2e.',
        workload_counts={'gather_192x6144': 79, 'moe_out_192x6144': 75, 'dense_out_192x6144': 3},
        errata=['AITER returns a separate output; SGLang gather writes it back at the callsite.',
                'The template false is is_broadcast_reg_outptr, not accumulate_template.']),
    'residual_rmsnorm_001': dict(template='rmsnorm', parent='residual_norm_quant_chain_001',
        kernel='measurement/kernel.py', nproc=1, fellow='triton-fellow', framework='aiter',
        operator='residual_add_rmsnorm', snapshot_aiter=True, targets=['fused_add_rmsnorm'],
        sources=['measurement/kernel.py'], editable=['measurement/kernel.py', 'measurement/candidate_'],
        scope='Residual-add RMSNorm component only, BF16 [48,6144]; excludes staging, collectives, full chain.',
        workload_counts={'residual_rmsnorm_48x6144': 156},
        errata=['FUSE_QUANT=false; epsilon=1e-5; normalize the unrounded FP32 residual sum, GLM weight offset zero.']),
    'dp_staging_001': dict(template='dp_staging', parent='residual_norm_quant_chain_001',
        kernel='measurement/kernel.py', nproc=1, fellow='triton-fellow', framework='sglang',
        operator='dp_gather_scatter_staging', snapshot_aiter=False,
        targets=['memcpy_triton_kernel'], sources=['measurement/kernel.py'],
        editable=['measurement/kernel.py', 'measurement/candidate_'],
        scope='Single-GPU local zero/copy components only, 79 gather + 78 scatter; no collective or norm.',
        workload_counts={'local_dp_staging_79g78s_48x6144': 1},
        errata=['Parent family share is not the component share. Preserve all zero-fill and padded-row semantics.']),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def git(repo, *args, check=True):
    result = subprocess.run(['git', '-C', str(repo), *args], capture_output=True, text=True, check=check)
    return result.stdout.strip()


def activate_forge(root):
    root = Path(root).resolve()
    for relative, expected in API_HASHES.items():
        if digest(root / relative) != expected:
            raise ValueError(f'KernelForge API version mismatch: {relative}; review adapter before running')
    sys.path.insert(0, str(root / 'src'))
    return root


def validate_analysis(analysis, kernelforge):
    validator = HERE.parent / 'analyze_profiles/tools/validate_handoff.py'
    spec = importlib.util.spec_from_file_location('forge_handoff_validator', validator)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate(analysis, kernelforge)


def snapshot_git(source, destination):
    """Copy pinned source content including submodules, without sharing Git state."""
    source, destination = Path(source), Path(destination)
    output = subprocess.check_output(['git', '-C', str(source), 'ls-files', '--recurse-submodules', '-z'])
    files = [os.fsdecode(f) for f in output.split(b'\0') if f]
    destination.mkdir(parents=True)
    hashes = {}
    for relative in files:
        src, dst = source / relative, destination / relative
        if not src.exists() and not src.is_symlink():
            continue  # Preserve tracked deletions as content absent from this snapshot.
        if src.is_dir():
            raise ValueError(f'Unexpanded submodule in snapshot: {src}')
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst, follow_symlinks=False)
        hashes[relative] = hashlib.sha256(os.readlink(src).encode()).hexdigest() if src.is_symlink() else digest(src)
    return {'root': str(source), 'commit': git(source, 'rev-parse', 'HEAD'),
            'dirty_diff': git(source, 'diff', 'HEAD'), 'submodules': git(source, 'submodule', 'status', '--recursive'),
            'files': hashes}


def bundle_manifest(bundle):
    bundle = Path(bundle).resolve()
    m = read_json(bundle / 'manifest.json')
    if m['bundle_path'] != str(bundle):
        raise ValueError('Bundle was moved; rebuild environment/path identity before resuming')
    return bundle, m


def protected_hashes(repo):
    return {str(p.relative_to(repo)): digest(p) for p in (repo / 'measurement').rglob('*')
            if p.is_file() and p.suffix not in {'.pyc'} and '__pycache__' not in p.parts
            and p.name != 'kernel.py' and not p.name.startswith('candidate_')}


def verify_protected(bundle, manifest):
    for relative, expected in manifest.get('protected_files', {}).items():
        path = bundle / 'repo' / relative
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'Protected measurement input changed: {relative}')
    for filename, identity in manifest.get('external_fixture_files', {}).items():
        path = Path(filename)
        if not path.is_file() or path.stat().st_size != identity['size'] or path.stat().st_mtime_ns != identity['mtime_ns']:
            raise ValueError(f'External fixture changed: {filename}')


def runtime_env(bundle, manifest):
    repo = bundle / 'repo'
    node = manifest.get('node', socket.gethostname())
    local = Path('/tmp/forge-runtime') / node / manifest['task_id']
    local.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    # Deliberately replace source search roots inherited from the serving environment.
    env.update(PYTHONPATH=os.pathsep.join([str(repo / 'aiter'), str(Path(manifest['kernelforge']) / 'src')]),
        PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
        TMPDIR=str(local), AITER_JIT_DIR=str(local / 'aiter'), TRITON_CACHE_DIR=str(local / 'triton'),
        TORCHINDUCTOR_CACHE_DIR=str(local / 'torch'), XDG_CACHE_HOME=str(local / 'xdg'),
        ROCPROF_TMPDIR=str(local / 'rocprofiler'),
        FORGE_ANALYSIS_MAX_ATTEMPTS=str(manifest.get('analysis_max_attempts', 2)),
        AITER_REBUILD='0', GPU_ARCHS='gfx950', GPU_TARGET='gfx950',
        HIP_VISIBLE_DEVICES=manifest['devices'], CUDA_VISIBLE_DEVICES=manifest['devices'],
        FORGE_NPROC_PER_NODE=str(manifest['recipe']['nproc']),
        DEBUG_CLR_GRAPH_PACKET_CAPTURE='false',
        GIT_AUTHOR_NAME='KernelForge experiment', GIT_AUTHOR_EMAIL='kernelforge@local',
        GIT_COMMITTER_NAME='KernelForge experiment', GIT_COMMITTER_EMAIL='kernelforge@local')
    for key in ('AITER_JIT_DIR', 'TRITON_CACHE_DIR', 'TORCHINDUCTOR_CACHE_DIR', 'XDG_CACHE_HOME', 'ROCPROF_TMPDIR'):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    if manifest['recipe']['snapshot_aiter']:
        env['FORGE_EXPECTED_AITER_ROOT'] = str(repo / 'aiter')
        env['FORGE_AITER_ROOT'] = str(repo / 'aiter')
    env.pop('AITER_QUICK_REDUCE_QUANTIZATION', None)
    return env


def verify_compat(manifest):
    root = Path(manifest['kernelforge'])
    receipt = read_json(root / '.forge_compat.json')
    if manifest.get('kernelforge_compat') != receipt:
        raise ValueError('KernelForge compatibility identity changed; use apply_compat.py on the stopped bundle')
    for name, expected in receipt['patches'].items():
        if digest(HERE / 'patches' / name) != expected:
            raise ValueError(f'Compatibility patch changed: {name}')
    for relative, expected in receipt['files'].items():
        if digest(root / relative) != expected:
            raise ValueError(f'KernelForge compatibility source changed: {relative}')


def load_gateway(env, credentials='/llm_gateway/credentials.json', config='/llm_gateway/config.toml'):
    """Map the existing mounted provider to Forge's OPENAI_* gateway contract."""
    env = env.copy()
    secrets = []
    if Path(credentials).is_file():
        values = read_json(credentials)
        env.update({str(k): str(v) for k, v in values.items()})
        secrets.extend(str(v) for v in values.values() if v)
    if Path(config).is_file():
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        cfg = tomllib.loads(Path(config).read_text())
        provider = cfg.get('model_providers', {}).get(cfg.get('model_provider'), {})
        env.setdefault('OPENAI_BASE_URL', provider.get('base_url', ''))
        key_name = provider.get('env_key', '')
        if key_name and env.get(key_name):
            env.setdefault('OPENAI_API_KEY', env[key_name])
        headers = dict(provider.get('http_headers', {}))
        for name, key in provider.get('env_http_headers', {}).items():
            if env.get(key):
                headers[name] = env[key]
        if headers:
            env.setdefault('OPENAI_CUSTOM_HEADERS', json.dumps(headers))
            secrets.extend(str(v) for v in headers.values() if v)
    if not env.get('OPENAI_BASE_URL') or not env.get('OPENAI_API_KEY'):
        raise ValueError('Forge requires OPENAI_BASE_URL and OPENAI_API_KEY; configure the mounted provider')
    secrets.append(env['OPENAI_API_KEY'])
    return env, sorted(set(secrets), key=len, reverse=True)


def redact(text, secrets):
    for secret in secrets:
        text = text.replace(secret, '[REDACTED]')
    return text


@contextlib.contextmanager
def environment(env):
    old = os.environ.copy()
    os.environ.clear()
    os.environ.update(env)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old)


@contextlib.contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Task/device lock is held: {path}') from exc
        yield


@contextlib.contextmanager
def device_locks(bundle, manifest):
    """Share device locks across all entry points and containers for this node."""
    devices = set(manifest['devices'].split(','))
    # Scripts are mounted at /run inside Docker, so HERE is not the project root.
    # Bundles retain the shared <forge-workspace>/<node>/<task> hierarchy.
    directory = bundle.parent.parent / '.gpu-locks'
    directory.mkdir(parents=True, exist_ok=True)
    node = manifest['node'].replace('/', '_')
    with contextlib.ExitStack() as stack:
        for device in sorted(devices):
            stack.enter_context(lock(directory / f'{node}-{device}.lock'))
        # Also recognize campaigns launched before shared device locks were added.
        # A stale running lease is deliberately not ignored; resolve its process
        # state before taking over a GPU after an unclean supervisor shutdown.
        for other in bundle.parent.iterdir():
            if other.resolve() == bundle.resolve() or not (other / 'run.json').is_file():
                continue
            state = read_json(other / 'run.json')
            if state.get('status') not in {'starting', 'running'}:
                continue
            identity = read_json(other / 'manifest.json')
            if identity['node'] == manifest['node'] and devices.intersection(identity['devices'].split(',')):
                raise RuntimeError(f'Device is leased by {other}; resolve that campaign before measuring')
        yield


def runtime_identity(env):
    code = ('import json,torch,importlib.metadata as m; '
            'print(json.dumps(dict(torch=torch.__version__,hip=torch.version.hip,'
            'gpu=torch.cuda.get_device_name(),gpu_count=torch.cuda.device_count(),'
            'arch=torch.cuda.get_device_properties(0).gcnArchName.split(\":\")[0],'
            'triton=m.version(\"triton\"))))')
    p = subprocess.run([sys.executable, '-c', code], env=env, capture_output=True, text=True, check=True)
    return json.loads(p.stdout.strip().splitlines()[-1])


def case_ids(cases):
    rows = cases['scored_cases']
    ids = [c['id'] for c in rows]
    if not ids or len(ids) != len(set(ids)) or any(not x or any(c.isspace() for c in x) for x in ids):
        raise ValueError('Cases must declare nonempty unique whitespace-free IDs')
    return ids
