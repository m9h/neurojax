#!/usr/bin/env python
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""WAND batch structural processing with dependency-aware load balancing.

Replaces fsl_sub shell-based submission with Python concurrent.futures
for local workstations. Runs independent tasks in parallel up to a
configurable CPU limit, respects task dependencies, and provides
real-time progress reporting.

Python processing scripts use PEP 723 inline metadata so each can be
launched via `uv run script.py` — separate cached venvs per script,
no shared env contention.

Usage:
    # Process single subject (auto-detects CPUs)
    uv run 30_batch_structural.py sub-08033

    # Multiple subjects with explicit workers
    uv run 30_batch_structural.py sub-08033 sub-09188 --workers 12

    # Dry run (show DAG, don't execute)
    uv run 30_batch_structural.py sub-08033 --dry-run

    # Resume (skip tasks whose output already exists)
    uv run 30_batch_structural.py sub-08033 --resume
"""
import argparse
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, Future, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("batch")


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


@dataclass
class Task:
    """A processing task with dependencies and resource requirements."""
    name: str
    command: list[str]
    depends_on: list[str] = field(default_factory=list)
    cpus: int = 1                # CPU cores this task needs
    output_marker: str = ""      # file path — if exists, task is done
    status: TaskStatus = TaskStatus.PENDING
    elapsed: float = 0.0
    error: str = ""


def run_task(task: Task) -> Task:
    """Execute a task in a subprocess. Called by ProcessPoolExecutor."""
    t0 = time.monotonic()
    try:
        result = subprocess.run(
            task.command,
            capture_output=True,
            text=True,
            timeout=7200,  # 2 hour max per task
        )
        task.elapsed = time.monotonic() - t0
        if result.returncode != 0:
            task.status = TaskStatus.FAILED
            # Keep last 500 chars of stderr for diagnosis
            task.error = (result.stderr or result.stdout or "")[-500:]
        else:
            task.status = TaskStatus.DONE
    except subprocess.TimeoutExpired:
        task.elapsed = time.monotonic() - t0
        task.status = TaskStatus.FAILED
        task.error = "Timed out after 2 hours"
    except Exception as e:
        task.elapsed = time.monotonic() - t0
        task.status = TaskStatus.FAILED
        task.error = str(e)
    return task


def build_structural_dag(subject: str, wand_root: str, script_dir: str,
                         resume: bool = False) -> list[Task]:
    """Build the task DAG for quantitative structural processing.

    Returns tasks in topological order. Independent tasks run in parallel;
    dependent tasks wait for their prerequisites.
    """
    deriv = f"{wand_root}/derivatives"
    fs_dir = f"{deriv}/freesurfer/{subject}"
    adv_dir = f"{deriv}/advanced-freesurfer/{subject}"
    qmri_dir = f"{deriv}/qmri/{subject}"
    anat_ses03 = f"{wand_root}/{subject}/ses-03/anat"

    # FreeSurfer + FSL environment setup
    fs_home = os.environ.get("FREESURFER_HOME", "/Applications/freesurfer/8.2.0")
    fsl_dir = os.environ.get("FSLDIR", "/Users/mhough/fsl")
    subjects_dir = f"{deriv}/freesurfer"

    env_prefix = (
        f"export FREESURFER_HOME={fs_home} && "
        f"source $FREESURFER_HOME/SetUpFreeSurfer.sh 2>/dev/null && "
        f"export SUBJECTS_DIR={subjects_dir} && "
        f"export FSLDIR={fsl_dir} && "
        f"source $FSLDIR/etc/fslconf/fsl.sh && "
    )

    def sh(cmd: str) -> list[str]:
        return ["bash", "-c", env_prefix + cmd]

    # Use `uv run` for PEP 723 scripts — each gets its own cached venv
    uv = "uv"

    tasks = []

    # --- Group 1: Independent tasks (all can run in parallel) ---

    # SAMSEG (multimodal T1+T2 segmentation)
    # T2w must be registered to T1w first (different FOV/resolution)
    t1w = f"{anat_ses03}/{subject}_ses-03_T1w.nii.gz"
    t2w_reg = f"{adv_dir}/myelin/T2w_in_T1w.nii.gz"
    samseg_dir = f"{adv_dir}/samseg"
    tasks.append(Task(
        name="samseg",
        command=sh(
            f"mkdir -p {samseg_dir} && "
            f"run_samseg --input {t1w} --input {t2w_reg} "
            f"--output {samseg_dir} --threads 2"
        ),
        cpus=2,
        output_marker=f"{samseg_dir}/seg.mgz",
    ))

    # Thalamic nuclei
    tasks.append(Task(
        name="thalamic_nuclei",
        command=sh(f"segmentThalamicNuclei.sh {subject} {subjects_dir}"),
        cpus=1,
        output_marker=f"{fs_dir}/mri/ThalamicNuclei.v13.T1.FSvoxelSpace.mgz",
    ))

    # Hippocampal subfields + amygdala
    tasks.append(Task(
        name="hippocampal_subfields",
        command=sh(
            f"segmentHA_T1.sh {subject} {subjects_dir} {t2w} T2"
        ),
        cpus=1,
        output_marker=f"{fs_dir}/mri/lh.hippoAmygLabels-T1-T2.v22.mgz",
    ))

    # Hypothalamic subunits (FS mode: saves to subject's mri/ dir)
    tasks.append(Task(
        name="hypothalamic_subunits",
        command=sh(
            f"mri_segment_hypothalamic_subunits "
            f"--s {subject} --sd {subjects_dir} --write_posteriors"
        ),
        cpus=1,
        output_marker=f"{fs_dir}/mri/hypothalamic_subunits_seg.v1.mgz",
    ))

    # WMH segmentation
    tasks.append(Task(
        name="wmh",
        command=sh(
            f"mkdir -p {adv_dir} && "
            f"mri_WMHsynthseg --i {t1w} "
            f"--o {adv_dir}/wmh_seg.nii.gz "
            f"--csv_vols {adv_dir}/wmh_volumes.csv"
        ),
        cpus=1,
        output_marker=f"{adv_dir}/wmh_seg.nii.gz",
    ))

    # SynthSeg across sessions
    synthseg_dir = f"{adv_dir}/synthseg"
    for ses in ["ses-02", "ses-03", "ses-04", "ses-05", "ses-06"]:
        ses_t1 = f"{wand_root}/{subject}/{ses}/anat/{subject}_{ses}_T1w.nii.gz"
        tasks.append(Task(
            name=f"synthseg_{ses}",
            command=sh(
                f"mkdir -p {synthseg_dir} && "
                f"[ -f {ses_t1} ] && "
                f"mri_synthseg --i {ses_t1} "
                f"--o {synthseg_dir}/{ses}_synthseg.nii.gz "
                f"--vol {synthseg_dir}/{ses}_volumes.csv "
                f"--qc {synthseg_dir}/{ses}_qc.csv --robust "
                f"|| echo 'No T1w for {ses}'"
            ),
            cpus=1,
            output_marker=f"{synthseg_dir}/{ses}_synthseg.nii.gz",
        ))

    # VFA T1 + QMT fitting (Python-based, uv run for isolated venv)
    tasks.append(Task(
        name="vfa_t1_qmt",
        command=[
            uv, "run",
            f"{script_dir}/29_qmri_vfa_qmt.py",
            subject,
            "--wand-root", wand_root,
        ],
        cpus=2,
        output_marker=f"{qmri_dir}/ses-02/T1map.nii.gz",
    ))

    # --- Group 2: Depends on VFA T1 + QMT ---

    # g-ratio + myelin proxy comparison
    tasks.append(Task(
        name="gratio_myelin",
        command=[
            uv, "run",
            f"{script_dir}/31_gratio_myelin_comparison.py",
            subject,
            "--wand-root", wand_root,
        ],
        depends_on=["vfa_t1_qmt"],
        cpus=1,
        output_marker=f"{qmri_dir}/gratio/g_ratio_proxy.nii.gz",
    ))

    # Apply resume logic
    if resume:
        for t in tasks:
            if t.output_marker and Path(t.output_marker).exists():
                t.status = TaskStatus.SKIPPED
                log.info(f"  SKIP (output exists): {t.name}")

    return tasks


class DAGExecutor:
    """Execute a task DAG with dependency-aware load balancing."""

    def __init__(self, tasks: list[Task], max_workers: int, max_cpus: int):
        self.tasks = {t.name: t for t in tasks}
        self.max_workers = max_workers
        self.max_cpus = max_cpus
        self.used_cpus = 0
        self.lock = Lock()
        self.futures: dict[str, Future] = {}

    def _ready(self, task: Task) -> bool:
        """Check if all dependencies are satisfied."""
        for dep_name in task.depends_on:
            dep = self.tasks.get(dep_name)
            if dep and dep.status not in (TaskStatus.DONE, TaskStatus.SKIPPED):
                return False
        return True

    def _can_schedule(self, task: Task) -> bool:
        """Check if there's CPU budget for this task."""
        return self.used_cpus + task.cpus <= self.max_cpus

    def run(self) -> dict[str, Task]:
        """Execute all tasks respecting dependencies and CPU limits."""
        pending = [
            t for t in self.tasks.values()
            if t.status == TaskStatus.PENDING
        ]
        total = len(pending)
        done_count = sum(
            1 for t in self.tasks.values()
            if t.status in (TaskStatus.DONE, TaskStatus.SKIPPED)
        )

        if total == 0:
            log.info("All tasks already complete!")
            return self.tasks

        log.info(f"Running {total} tasks ({done_count} already done/skipped)")
        log.info(f"Max workers: {self.max_workers}, Max CPUs: {self.max_cpus}")

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            active: dict[Future, str] = {}

            while True:
                # Submit ready tasks that fit within CPU budget
                for name, task in self.tasks.items():
                    if task.status != TaskStatus.PENDING:
                        continue
                    if not self._ready(task):
                        continue
                    if not self._can_schedule(task):
                        continue
                    if name in active.values():
                        continue

                    task.status = TaskStatus.RUNNING
                    self.used_cpus += task.cpus
                    future = executor.submit(run_task, task)
                    active[future] = name
                    log.info(f"  START: {name} (cpus={task.cpus}, "
                             f"used={self.used_cpus}/{self.max_cpus})")

                if not active:
                    # Nothing running — check if we're blocked or done
                    remaining = [
                        t for t in self.tasks.values()
                        if t.status == TaskStatus.PENDING
                    ]
                    if not remaining:
                        break
                    # Check for deadlock
                    blocked = all(
                        not self._ready(t) for t in remaining
                    )
                    if blocked:
                        failed_deps = set()
                        for t in remaining:
                            for dep in t.depends_on:
                                if self.tasks[dep].status == TaskStatus.FAILED:
                                    failed_deps.add(dep)
                        if failed_deps:
                            log.error(f"Deadlocked: failed dependencies {failed_deps}")
                            for t in remaining:
                                t.status = TaskStatus.FAILED
                                t.error = f"Blocked by failed: {failed_deps}"
                            break
                    continue

                # Wait for at least one task to finish
                done_futures = []
                for future in list(active.keys()):
                    if future.done():
                        done_futures.append(future)

                if not done_futures:
                    # Brief sleep then re-check
                    time.sleep(0.5)
                    continue

                for future in done_futures:
                    name = active.pop(future)
                    result_task = future.result()
                    self.tasks[name] = result_task
                    self.used_cpus -= result_task.cpus

                    if result_task.status == TaskStatus.DONE:
                        done_count += 1
                        log.info(
                            f"  DONE:  {name} "
                            f"({result_task.elapsed:.0f}s) "
                            f"[{done_count}/{total + done_count - len(pending) + sum(1 for t in self.tasks.values() if t.status == TaskStatus.SKIPPED)}]"
                        )
                    elif result_task.status == TaskStatus.FAILED:
                        log.error(
                            f"  FAIL:  {name} "
                            f"({result_task.elapsed:.0f}s): "
                            f"{result_task.error[:200]}"
                        )

        return self.tasks


def print_dag(tasks: list[Task]):
    """Print the task DAG for visualization."""
    print("\nTask DAG:")
    print("=" * 60)
    for t in tasks:
        deps = f" (after: {', '.join(t.depends_on)})" if t.depends_on else ""
        status = t.status.value
        marker = " [EXISTS]" if t.output_marker and Path(t.output_marker).exists() else ""
        print(f"  [{status:>7s}] {t.name:<30s} cpus={t.cpus}{deps}{marker}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="WAND batch structural processing with load balancing"
    )
    parser.add_argument("subjects", nargs="+", help="Subject IDs")
    parser.add_argument("--wand-root", default="/Users/mhough/dev/wand")
    parser.add_argument("--workers", type=int, default=6,
                        help="Max concurrent processes")
    parser.add_argument("--cpus", type=int, default=0,
                        help="Max total CPUs (0 = auto-detect)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show DAG without executing")
    parser.add_argument("--resume", action="store_true",
                        help="Skip tasks whose output already exists")
    args = parser.parse_args()

    if args.cpus <= 0:
        args.cpus = os.cpu_count() or 8

    script_dir = str(Path(__file__).parent)

    for subject in args.subjects:
        log.info(f"\n{'='*60}")
        log.info(f"Subject: {subject}")
        log.info(f"{'='*60}")

        tasks = build_structural_dag(
            subject, args.wand_root, script_dir, resume=args.resume
        )
        print_dag(tasks)

        if args.dry_run:
            continue

        executor = DAGExecutor(tasks, args.workers, args.cpus)
        t0 = time.monotonic()
        results = executor.run()
        wall = time.monotonic() - t0

        # Summary
        print(f"\n{'='*60}")
        print(f"Subject: {subject} — completed in {wall:.0f}s")
        for status in TaskStatus:
            count = sum(1 for t in results.values() if t.status == status)
            if count:
                print(f"  {status.value}: {count}")
        failed = [t for t in results.values() if t.status == TaskStatus.FAILED]
        if failed:
            print("\nFailed tasks:")
            for t in failed:
                print(f"  {t.name}: {t.error[:200]}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
