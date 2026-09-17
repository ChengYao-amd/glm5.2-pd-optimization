#!/usr/bin/env python3
"""Apply the compatibility patch to a temporary source copy and test CPU guards.

Usage: python3 run/run_forge/patches/test_analysis_artifacts.py
No installed source is modified; no SDK request, credentials, or GPU is used.
"""
from __future__ import annotations

import argparse
import ast
from dataclasses import replace
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


def git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


class ArtifactGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="forge-artifact-guard-")
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / "repo"
        self.repo.mkdir()
        git(self.repo, "init", "-q")
        git(self.repo, "config", "user.name", "Guard regression")
        git(self.repo, "config", "user.email", "guard@example.invalid")
        (self.repo / ".gitignore").write_text("forge_experiments/\nignored.txt\n")
        self.kernel = self.repo / "kernel.py"
        self.driver = self.repo / "driver.py"
        self.kernel.write_text("VALUE = 1\n")
        self.driver.write_text("ORACLE = 1\n")
        git(self.repo, "add", ".")
        git(self.repo, "commit", "-qm", "baseline")
        self.work = self.repo / "forge_experiments/analysis/work/commit"
        self.work.mkdir(parents=True)
        self.report = self.work / "report.md"
        self.report.write_text("initial report\n")
        self.spec = AgentRunSpec(
            system_prompt="Analyze", user_prompt="Profile", cwd=str(self.work),
            writable=True, protected_globs=["*"], driver_script=str(self.driver),
            writable_artifact_roots=[str(self.work)], allow_untracked=True,
        )

    def guard(self, spec=None):
        guard = _WorkspaceGuard(spec or self.spec)
        guard.prepare()
        self.addCleanup(guard.rollback)
        return guard

    def test_ignored_reports_profiles_and_existing_artifacts_are_writable(self):
        guard = self.guard()
        self.assertNotIn(self.report, guard.baseline_protected_ignored)
        self.report.write_text("updated report\n")
        (self.work / "commands.jsonl").write_text('{"command":"profile"}\n')
        (self.work / "profile").mkdir()
        (self.work / "profile/counters.csv").write_text("metric,value\ncycles,1\n")
        self.assertEqual(guard.verify(), [])
        self.assertEqual(self.report.read_text(), "updated report\n")

    def test_nonignored_artifacts_are_writable_without_global_untracked_permission(self):
        work = self.repo / "analysis-output"
        work.mkdir()
        guard = self.guard(replace(self.spec, cwd=str(work),
                                  writable_artifact_roots=[str(work)], allow_untracked=False))
        (work / "report.md").write_text("new report\n")
        self.assertEqual(guard.verify(), [])

    def test_original_failure_remains_without_explicit_opt_in(self):
        guard = self.guard(replace(self.spec, writable_artifact_roots=[]))
        self.report.write_text("changed\n")
        with self.assertRaises(CodexSafetyError):
            guard.verify()
        self.assertEqual(self.report.read_text(), "initial report\n")

    def test_source_edit_is_rejected_and_restored(self):
        guard = self.guard()
        self.kernel.write_text("VALUE = 99\n")
        with self.assertRaises(CodexSafetyError):
            guard.verify()
        self.assertEqual(self.kernel.read_text(), "VALUE = 1\n")

    def test_driver_edit_is_rejected_and_restored(self):
        guard = self.guard()
        self.driver.write_text("ORACLE = 99\n")
        with self.assertRaises(CodexSafetyError):
            guard.verify()
        self.assertEqual(self.driver.read_text(), "ORACLE = 1\n")

    def test_ignored_file_outside_artifacts_stays_protected(self):
        outside = self.repo / "ignored.txt"
        outside.write_text("original\n")
        guard = self.guard()
        outside.write_text("modified\n")
        with self.assertRaises(CodexSafetyError):
            guard.verify()
        self.assertEqual(outside.read_text(), "original\n")

    def test_new_source_outside_artifacts_is_rejected(self):
        guard = self.guard()
        (self.repo / "new_source.py").write_text("VALUE = 1\n")
        with self.assertRaises(CodexSafetyError):
            guard.verify()

    def test_explicit_protected_file_inside_artifacts_stays_protected(self):
        guard = self.guard(replace(self.spec, protected_paths=[str(self.report)]))
        self.report.write_text("modified\n")
        with self.assertRaises(CodexSafetyError):
            guard.verify()
        self.assertEqual(self.report.read_text(), "initial report\n")

    def test_explicit_protected_file_outside_repository_stays_protected(self):
        outside = Path(self.tmp.name) / "external_oracle.py"
        outside.write_text("ORACLE = 1\n")
        guard = self.guard(replace(self.spec, protected_paths=[str(outside)]))
        outside.write_text("ORACLE = 99\n")
        with self.assertRaises(CodexSafetyError):
            guard.verify()
        self.assertEqual(outside.read_text(), "ORACLE = 1\n")

    def test_artifact_symlink_escape_is_rejected(self):
        guard = self.guard()
        (self.work / "escape").symlink_to(self.kernel)
        with self.assertRaises(CodexSafetyError):
            guard.verify()

    def test_staging_artifacts_remains_forbidden(self):
        guard = self.guard()
        git(self.repo, "add", "-f", str(self.report))
        with self.assertRaises(CodexSafetyError):
            guard.verify()

    def test_head_change_remains_forbidden(self):
        guard = self.guard()
        git(self.repo, "commit", "--allow-empty", "-qm", "forbidden commit")
        with self.assertRaises(CodexSafetyError):
            guard.verify()

    def test_broad_source_and_git_roots_are_rejected(self):
        for path in (self.repo, self.repo / ".git", self.repo / "elsewhere"):
            with self.subTest(path=path):
                with self.assertRaises(CodexSafetyError):
                    _WorkspaceGuard(replace(self.spec, writable_artifact_roots=[str(path)])).prepare()

    def test_tracked_files_inside_declared_root_are_rejected(self):
        git(self.repo, "add", "-f", str(self.report))
        git(self.repo, "commit", "-qm", "tracked file in output directory")
        with self.assertRaises(CodexSafetyError):
            _WorkspaceGuard(self.spec).prepare()

    def test_read_only_specialist_still_skips_guard_without_artifact_opt_in(self):
        spec = AgentRunSpec(system_prompt="Read", user_prompt="Inspect", cwd=self.tmp.name,
                            writable=False, protected_globs=["*"],
                            tool_policy=AgentToolPolicy(write=False, shell=False))
        guard = _WorkspaceGuard(spec)
        guard.prepare()
        self.assertTrue(guard.skipped)
        self.assertEqual(guard.verify(), [])

    def test_resolved_spec_preserves_artifact_contract(self):
        resolved = self.spec.resolved(AgentRuntimeConfig(provider="codex", model="test"))
        self.assertEqual(resolved.writable_artifact_roots, [str(self.work)])

    def test_analysis_declares_only_work_root_as_artifact_root(self):
        tree = ast.parse((PATCHED_ROOT / "src/kernel_agents/orchestrator/analysis.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "AgentRunSpec"]
        declarations = [kw.value for node in calls for kw in node.keywords
                        if kw.arg == "writable_artifact_roots"]
        self.assertEqual(len(declarations), 1)
        self.assertEqual(ast.dump(declarations[0]), ast.dump(ast.parse("[str(work_root)]", mode="eval").body))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path,
                        default=Path(__file__).resolve().parents[3] / "third-party/KernelForge")
    args, remaining = parser.parse_known_args()
    with tempfile.TemporaryDirectory(prefix="forge-patched-source-") as tmp:
        PATCHED_ROOT = Path(tmp)
        # Only a temporary source copy is mutated. No credentials or runtime
        # configuration are copied or evaluated.
        shutil.copytree(args.upstream / "src", PATCHED_ROOT / "src",
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        subprocess.run(["git", "apply", str(Path(__file__).with_name("analysis-artifacts.patch").resolve())],
                       cwd=PATCHED_ROOT, check=True, capture_output=True, text=True)
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(PATCHED_ROOT / "src"))
        from forge_llm.agent_backends.base import AgentRunSpec, AgentRuntimeConfig, AgentToolPolicy
        from forge_llm.agent_backends.codex import _WorkspaceGuard, CodexSafetyError
        unittest.main(argv=[sys.argv[0], *remaining])
