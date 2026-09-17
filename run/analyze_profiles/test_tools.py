"""Contract/streaming edge cases; real SDK/GPU tests live in the experiment record."""

import gzip
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

from ijson.common import JSONError
import yaml

from tools.trace_io import iter_events
from tools.validate_handoff import validate


class TraceTests(unittest.TestCase):
    def test_summary_includes_events_without_category(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'trace.json'
            path.write_text(json.dumps({'traceEvents': [{'ph': 'M'}] * 8 + [{'cat': 'kernel'}]}))
            script = Path(__file__).parent / 'tools/trace_io.py'
            result = subprocess.run([sys.executable, str(script), str(path)], check=True, capture_output=True, text=True)
            self.assertEqual(json.loads(result.stdout)['events'], 9)

    def test_plain_and_gzip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payload = json.dumps({'schemaVersion': 1, 'traceEvents': [
                {'cat': 'kernel', 'name': '核', 'args': {'grid': [1, 2, 3]}},
                {'cat': 'cpu_op', 'dur': 1.25}], 'trailer': True}).encode()
            for name in ['trace.json', 'trace.json.gz']:
                path = root / name
                path.write_bytes(gzip.compress(payload) if name.endswith('gz') else payload)
                events = list(iter_events(path))
                self.assertEqual(events[0]['args']['grid'], [1, 2, 3])
                self.assertEqual(events[1]['dur'], 1.25)

    def test_missing_and_truncated_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'trace.json'
            for text in ['{}', '{"traceEvents": [{"name": "x"}', '{"traceEvents": []']:
                path.write_text(text)
                with self.assertRaises((ValueError, JSONError)):
                    list(iter_events(path))
            path.write_text('{"traceEvents": []}')
            self.assertEqual(list(iter_events(path)), [])


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        registry = self.root / 'forge/src/kernel_agents/fellows/constants.py'
        registry.parent.mkdir(parents=True)
        registry.write_text("FELLOW_AGENT_MODULES = {'triton': 'module'}\n")
        self.forge = self.root / 'forge'
        (self.root / 'tasks').mkdir()
        (self.root / 'source.py').write_text('def op(x): return x\n')
        (self.root / 'evidence.txt').write_text('source and trace evidence\n')
        (self.root / 'candidates.yaml').write_text(yaml.safe_dump({
            'candidates': [{'id': 'op', 'name': 'op', 'task': 'tasks/op.yaml'}]}))
        self.task = {
            'task_id': 'op', 'description': 'Optimize op', 'operation': 'op',
            'dtype': 'bf16', 'gpu_target': 'gfx950', 'backends': ['triton'],
            'shapes': {'primary': {'M': 8}, 'validation': []},
            'kernels_to_review': [{'kernel': 'op', 'source_path': 'source.py',
                'bottleneck_hypothesis': 'extra copy', 'tunable_surfaces': ['fusion']}],
            'constraints': [], 'validation_gates': ['check all outputs'],
            'priority': 'P1', 'status': 'pending',
            'x-handoff': {
                'schema_version': 1, 'kind': 'analysis', 'readiness': 'ready_for_handoff',
                'driver_owner': 'kernelforge', 'workload': {}, 'source': {'revision': 'test'},
                'inputs': [{'name': 'x', 'knowledge': 'observed', 'spec': {'shape': [8]}, 'evidence_refs': ['E1']}],
                'measurements': [{'name': 'latency', 'value': 2.3, 'unit': 'us', 'scope': 'rank0', 'evidence_refs': ['E1']}],
                'roofline': {'status': 'estimated', 'explanation': 'logical bytes', 'evidence_refs': ['E1']},
                'evidence': [{'id': 'E1', 'path': 'evidence.txt', 'locator': 'line 1'}],
                'missing_information': [],
            },
        }

    def check(self):
        (self.root / 'tasks/op.yaml').write_text(yaml.safe_dump(self.task))
        return validate(self.root, self.forge)[0]

    def test_valid_analysis_needs_no_driver(self):
        self.assertEqual(self.check(), [])

    def test_unknown_source_cannot_be_ready(self):
        self.task['kernels_to_review'][0]['source_path'] = None
        self.assertTrue(any('unresolved kernel' in e for e in self.check()))

    def test_unknown_backend_and_missing_evidence(self):
        self.task['backends'] = ['tilelang']
        self.task['x-handoff']['evidence'][0]['path'] = 'missing.txt'
        errors = self.check()
        self.assertTrue(any('unsupported backend' in e for e in errors))
        self.assertTrue(any('evidence file missing' in e for e in errors))

    def test_nonfinite_measurement_and_missing_candidate(self):
        self.task['x-handoff']['measurements'][0]['value'] = float('nan')
        self.assertTrue(any('finite' in e for e in self.check()))
        (self.root / 'candidates.yaml').write_text('candidates: [{id: missing, task: tasks/missing.yaml}]')
        self.assertTrue(validate(self.root, self.forge)[0])

    def test_merge_cycle(self):
        (self.root / 'candidates.yaml').write_text('candidates: [{id: a, merged_into: b}, {id: b, merged_into: a}]')
        self.assertTrue(any('cyclic' in e for e in validate(self.root, self.forge)[0]))


if __name__ == '__main__':
    unittest.main()
