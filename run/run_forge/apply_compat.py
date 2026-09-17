#!/usr/bin/env python3
"""Apply audited compatibility patches to a dedicated KernelForge installation."""
import argparse
from pathlib import Path
import subprocess
import time

from lib.common import HERE, activate_forge, digest, read_json, write_json

PATCHES = ['analysis-artifacts.patch', 'analysis-retry-budget.patch']
FILES = ['src/forge_llm/agent_backends/base.py', 'src/forge_llm/agent_backends/codex.py',
         'src/kernel_agents/orchestrator/analysis.py',
         'src/kernel_agents/orchestrator/analysis_session.py']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kernelforge', type=Path, default=Path('/KernelForge'))
    p.add_argument('--bundle', type=Path, help='Record compatible engine identity on an existing stopped bundle')
    p.add_argument('--extra-analysis-retry', action='store_true', help='Allow attempt 3 without resetting prior attempt history')
    args = p.parse_args()
    root = activate_forge(args.kernelforge)
    if args.bundle:
        path = args.bundle / 'run.json'
        if path.exists() and read_json(path).get('status') in {'running', 'starting'}:
            raise ValueError('Stop the campaign before recording compatibility changes')
    before = {f: digest(root / f) for f in FILES}
    for name in PATCHES:
        patch = HERE / 'patches' / name
        reverse = subprocess.run(['git', 'apply', '--reverse', '--check', str(patch)], cwd=root,
                                 capture_output=True)
        if reverse.returncode == 0:
            continue
        subprocess.run(['git', 'apply', '--check', str(patch)], cwd=root, check=True)
        subprocess.run(['git', 'apply', str(patch)], cwd=root, check=True)
    receipt = {'patches': {n: digest(HERE / 'patches' / n) for n in PATCHES},
               'files': {f: digest(root / f) for f in FILES}}
    write_json(root / '.forge_compat.json', receipt)
    if args.bundle:
        m = read_json(args.bundle / 'manifest.json')
        m['kernelforge_compat'] = receipt
        if args.extra_analysis_retry:
            m['analysis_max_attempts'] = 3
        write_json(args.bundle / 'artifacts/compatibility-upgrade.json',
                   {'timestamp': time.time(), 'before': before, 'after': receipt,
                    'reason': 'Permit declared Analysis artifacts while retaining source/driver/Git protection',
                    'analysis_max_attempts': m.get('analysis_max_attempts', 2)})
        write_json(args.bundle / 'manifest.json', m)
    print('Compatibility patches verified; source/driver protection remains enabled')


if __name__ == '__main__':
    main()
