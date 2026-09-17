"""CPU checks for boundaries that protect an expensive optimization campaign."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from lib.common import RECIPES, case_ids, device_locks, digest, verify_protected
from run import command, remaining_budget, startup_owner_alive
from collect import kept_revision
from prepare import load_recipe


class ContractTests(unittest.TestCase):
    def test_custom_recipe_boundaries(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'recipe.yaml'
            recipe = dict(RECIPES['tp4_bf16_allreduce_001'], nproc=8,
                          workload='TP8 DP8 EP1; global verify M768, local M96')
            path.write_text(yaml.safe_dump(recipe))
            self.assertEqual(load_recipe('tp8', path)['nproc'], 8)
            for change in ({'kernel': '/aiter/original.py'}, {'editable': ['../outside']},
                           {'nproc': 0}, {'workload': ''}, {'sources': []}):
                path.write_text(yaml.safe_dump(dict(recipe, **change)))
                with self.assertRaises(ValueError):
                    load_recipe('tp8', path)

    def test_case_identity(self):
        for rows in ([], [{'id': 'x'}, {'id': 'x'}], [{'id': 'white space'}]):
            with self.assertRaises(ValueError):
                case_ids({'scored_cases': rows})
        self.assertEqual(case_ids({'scored_cases': [{'id': 'a'}, {'id': 'b'}]}), ['a', 'b'])

    def test_reference_mutation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / 'repo/measurement/reference.py'
            p.parent.mkdir(parents=True)
            p.write_text('fixed oracle')
            m = {'protected_files': {'measurement/reference.py': digest(p)}}
            verify_protected(root, m)
            p.write_text('modified oracle')
            with self.assertRaises(ValueError):
                verify_protected(root, m)

    def test_resume_budget(self):
        ledger = {'budget_hours': 12, 'sessions': [{'elapsed_seconds': 8 * 3600}]}
        self.assertEqual(remaining_budget(ledger, 12), 4 * 3600)
        with self.assertRaises(ValueError):
            remaining_budget(ledger, 24)

    def test_argv_is_not_shell(self):
        m = dict(recipe=RECIPES['tp4_bf16_allreduce_001'], model='model', effort='max',
                 bench_repeat=3, snr_threshold=60)
        root = Path('/tmp/task path; not shell')
        argv = command(root, m, 12, 1234)
        self.assertEqual(argv[argv.index('--workspace') + 1], str(root / 'repo'))
        self.assertEqual(argv[argv.index('--nproc-per-node') + 1], '4')
        resumed = command(root, m, 4, 5678, resume=True)
        self.assertIn('--resume', resumed)
        self.assertNotIn('--driver', resumed)

    def test_select_kept_not_working_head(self):
        self.assertEqual(kept_revision({'best': {'commit_hash': 'kept'}}, 'pristine'), 'kept')
        self.assertEqual(kept_revision({'cumulative': {'kept': 0}}, 'pristine'), 'pristine')
        with self.assertRaises(ValueError):
            kept_revision({'cumulative': {'kept': 1}}, 'pristine')

    def test_failed_start_owner_audit(self):
        with patch('run.socket.gethostname', return_value='same'), patch('run.os.kill') as kill:
            self.assertFalse(startup_owner_alive({'host': 'same'}))
            self.assertTrue(startup_owner_alive({'host': 'same', 'supervisor_pid': 42}))
            kill.side_effect = ProcessLookupError
            self.assertFalse(startup_owner_alive({'host': 'same', 'supervisor_pid': 42}))
            self.assertFalse(startup_owner_alive({'host': 'different', 'supervisor_pid': 42}))

    def test_device_lease_blocks_other_bundle(self):
        with tempfile.TemporaryDirectory() as tmp, patch('lib.common.ROOT', Path(tmp)):
            parent = Path(tmp) / 'workspace/forge-workspace/025'
            first, second = parent / 'first', parent / 'second'
            first.mkdir(parents=True)
            second.mkdir()
            m = {'node': '025', 'devices': '0,1'}
            (second / 'manifest.json').write_text(json.dumps(m))
            (second / 'run.json').write_text(json.dumps({'status': 'running'}))
            with self.assertRaises(RuntimeError):
                with device_locks(first, m):
                    pass
            (second / 'run.json').write_text(json.dumps({'status': 'finished'}))
            with device_locks(first, m):
                with self.assertRaises(RuntimeError):
                    with device_locks(first, m):
                        pass


if __name__ == '__main__':
    unittest.main()
