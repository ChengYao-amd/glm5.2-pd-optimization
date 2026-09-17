#!/usr/bin/env python3
"""Check handoff readiness or revalidate an existing prepared driver bundle."""
import argparse
import json
from pathlib import Path

from lib.common import (RECIPES, activate_forge, bundle_manifest, device_locks, lock, runtime_env,
                        validate_analysis, verify_protected, write_json)
from lib.forge_api import preflight


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-dir', type=Path)
    p.add_argument('--kernelforge', type=Path, default=Path('/KernelForge'))
    p.add_argument('--driver-bundle', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    activate_forge(args.kernelforge)
    if args.driver_bundle:
        bundle, m = bundle_manifest(args.driver_bundle)
        with lock(bundle / 'operation.lock'), device_locks(bundle, m):
            verify_protected(bundle, m)
            result = preflight(bundle, m, runtime_env(bundle, m))
    else:
        if not args.analysis_dir:
            p.error('--analysis-dir or --driver-bundle is required')
        errors, tasks = validate_analysis(args.analysis_dir, args.kernelforge)
        result = {'valid': not errors, 'errors': errors, 'tasks': tasks, 'recipes': list(RECIPES)}
    if args.output:
        write_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result.get('valid', result.get('ok', True)):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
