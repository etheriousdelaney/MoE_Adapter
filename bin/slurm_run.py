#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


USAGE = "usage: slurm_run.py [options] [JOB=1:n] log-file command-line arguments..."


@dataclass
class SlurmOptions:
    account: str
    partition: str
    gpus: int
    cpus_per_task: int
    time_limit: Optional[str]
    mem: Optional[str]
    max_jobs_run: int
    job_pick: str


@dataclass
class RunningJob:
    job_id: int
    log_path: str
    process: subprocess.Popen


def shell_quote(arg: str) -> str:
    return shlex.quote(arg)


def build_command(args: List[str]) -> str:
    return " ".join(shell_quote(arg) for arg in args) + " "


def format_now() -> str:
    return time.strftime("%a %b %d %H:%M:%S %Z %Y", time.localtime())


def ensure_parent_dir(logfile: str) -> None:
    Path(logfile).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


def write_log_header(logfile: str, cmd: str, srun_cmd: list[str]) -> int:
    ensure_parent_dir(logfile)
    start_time = int(time.time())
    with open(logfile, "w", encoding="utf-8") as handle:
        handle.write(f"# {cmd}\n")
        handle.write(f"# srun: {' '.join(shell_quote(arg) for arg in srun_cmd)}\n")
        handle.write(f"# Started at {format_now()}\n")
        handle.write("#\n")
    return start_time


def append_log_footer(logfile: str, returncode: int, start_time: int) -> None:
    end_time = int(time.time())
    return_str = f"code {returncode}" if returncode >= 0 else f"code 0; signal {-returncode}"
    with open(logfile, "a", encoding="utf-8") as handle:
        handle.write(f"# Accounting: time={end_time - start_time} threads=1\n")
        handle.write(
            f"# Ended ({return_str}) at {format_now()}, elapsed time {end_time - start_time} seconds\n"
        )


def inspect_log_for_pick(logfile: str, job_pick: str) -> str:
    if job_pick == "all":
        return "run"
    try:
        with open(logfile, "r", encoding="utf-8", errors="replace") as handle:
            log_line = None
            for cur_line in handle:
                if re.match(r"# Ended \(code .*", cur_line):
                    log_line = cur_line
    except OSError:
        return "run"
    if log_line is None:
        return "run"
    if re.match(r"# Ended \(code 0\).*", log_line):
        return "skip_success"
    if re.match(r"# Ended \(code \d+(; signal \d+)?\).*", log_line):
        return "run" if job_pick in {"failed", "all"} else "skip_failure"
    return "run"


def build_srun_command(options: SlurmOptions, cmd: str) -> list[str]:
    srun_cmd = [
        "srun",
        "--partition",
        options.partition,
        "--ntasks",
        "1",
        "--cpus-per-task",
        str(options.cpus_per_task),
    ]
    if options.account:
        srun_cmd.extend(["--account", options.account])
    if options.gpus > 0:
        srun_cmd.extend(["--gpus-per-node", str(options.gpus)])
    if options.time_limit:
        srun_cmd.extend(["--time", options.time_limit])
    if options.mem:
        srun_cmd.extend(["--mem", options.mem])
    srun_cmd.extend(["bash", "-lc", f"( {cmd})"])
    return srun_cmd


def launch_job(
    job_id: int,
    jobname: Optional[str],
    logfile: str,
    cmd: str,
    options: SlurmOptions,
) -> Tuple[Optional[RunningJob], Optional[int]]:
    if jobname is not None:
        cmd = cmd.replace(jobname, str(job_id))
        logfile = logfile.replace(jobname, str(job_id))

    pick_action = inspect_log_for_pick(logfile, options.job_pick)
    if pick_action == "skip_success":
        return None, 0
    if pick_action == "skip_failure":
        return None, 1

    srun_cmd = build_srun_command(options, cmd)
    start_time = write_log_header(logfile, cmd, srun_cmd)
    log_handle = open(logfile, "a", encoding="utf-8")
    process = subprocess.Popen(
        srun_cmd,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    setattr(process, "_slurm_run_log_handle", log_handle)
    setattr(process, "_slurm_run_start_time", start_time)
    return RunningJob(job_id=job_id, log_path=logfile, process=process), None


def finalize_job(running_job: RunningJob) -> int:
    process = running_job.process
    returncode = process.wait()
    log_handle = getattr(process, "_slurm_run_log_handle")
    log_handle.close()
    start_time = getattr(process, "_slurm_run_start_time")
    append_log_footer(running_job.log_path, returncode, start_time)
    return 0 if returncode == 0 else 1


def reap_one(active_jobs: Dict[int, RunningJob]) -> Tuple[int, int]:
    while True:
        for pid, running_job in list(active_jobs.items()):
            if running_job.process.poll() is None:
                continue
            code = finalize_job(running_job)
            del active_jobs[pid]
            return running_job.job_id, code
        time.sleep(0.2)


def parse_positive_int(value: str, name: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise SystemExit(f"slurm_run.py: {name} must be positive, got {value}")
    return parsed


def parse_args(argv: List[str]) -> Tuple[str, int, int, Optional[str], List[str], str, SlurmOptions]:
    if argv and argv[0] in {"-h", "--help"}:
        print(USAGE)
        raise SystemExit(0)
    if len(argv) < 2:
        raise SystemExit(USAGE)

    args = list(argv)
    job_pick = "all"
    max_jobs_run = -1
    jobstart = 1
    jobend = 1
    jobname = None
    account = os.environ.get("NANO_ACCOUNT", "")
    partition = os.environ.get("NANO_PARTITION", "normal")
    gpus = int(os.environ.get("NANO_GPUS", "1"))
    cpus_per_task = int(os.environ.get("NANO_CPUS_PER_TASK", "1"))
    time_limit = os.environ.get("NANO_TIME")
    mem = os.environ.get("NANO_MEM")

    for _ in range(3):
        while len(args) >= 2 and args[0].startswith("-"):
            switch = args.pop(0)
            if switch == "-V":
                continue
            if switch in {"--max-jobs-run", "-tc"}:
                new_constraint = parse_positive_int(args.pop(0), switch)
                max_jobs_run = new_constraint if max_jobs_run <= 0 else min(max_jobs_run, new_constraint)
                continue
            if not args:
                raise SystemExit(f"slurm_run.py: missing argument to {switch}")
            argument = args.pop(0)
            if argument.startswith("--"):
                print(
                    f"slurm_run.py: WARNING: suspicious argument '{argument}' to {switch}; starts with '-'",
                    file=sys.stderr,
                )
            if switch == "-sync":
                continue
            if switch == "-pe":
                if not args:
                    raise SystemExit("slurm_run.py: missing second argument to -pe")
                cpus_per_task = parse_positive_int(args.pop(0), "-pe")
                continue
            if switch in {"--gpu", "--gpus"}:
                gpus = int(argument)
                continue
            if switch == "--num-threads":
                cpus_per_task = parse_positive_int(argument, "--num-threads")
                continue
            if switch == "--time":
                time_limit = argument
                continue
            if switch == "--mem":
                mem = None if argument in {"", "0"} else argument
                continue
            if switch == "--account":
                account = argument
                continue
            if switch == "--partition":
                partition = argument
                continue
            if switch == "--pick":
                if argument not in {"all", "failed", "incomplete"}:
                    raise SystemExit(
                        "slurm_run.py: ERROR: --pick argument must be one of 'all', 'failed' or 'incomplete'"
                    )
                job_pick = argument
                continue
            raise SystemExit(f"slurm_run.py: unsupported option {switch}")

        if args and re.match(r"^([\w_][\w\d_]*)=(\d+):(\d+)$", args[0]):
            match = re.match(r"^([\w_][\w\d_]*)=(\d+):(\d+)$", args.pop(0))
            assert match is not None
            jobname = match.group(1)
            jobstart = int(match.group(2))
            jobend = int(match.group(3))
            if jobstart > jobend or jobstart <= 0:
                raise SystemExit(f"slurm_run.py: invalid job range {jobname}={jobstart}:{jobend}")
        elif args and re.match(r"^([\w_][\w\d_]*)=(\d+)$", args[0]):
            match = re.match(r"^([\w_][\w\d_]*)=(\d+)$", args.pop(0))
            assert match is not None
            jobname = match.group(1)
            jobstart = int(match.group(2))
            jobend = jobstart
        elif args and re.match(r".+=.*:.*$", args[0]):
            print(f"slurm_run.py: Warning: suspicious first argument: {args[0]}", file=sys.stderr)

    if len(args) < 2:
        raise SystemExit(USAGE)
    if not partition:
        raise SystemExit("slurm_run.py: NANO_PARTITION or --partition is required")
    if not account and "SLURM_JOB_ID" not in os.environ:
        raise SystemExit("slurm_run.py: NANO_ACCOUNT or --account is required outside an existing Slurm job")
    if max_jobs_run == -1:
        max_jobs_run = 1

    logfile = args.pop(0)
    options = SlurmOptions(
        account=account,
        partition=partition,
        gpus=gpus,
        cpus_per_task=cpus_per_task,
        time_limit=time_limit,
        mem=mem,
        max_jobs_run=max_jobs_run,
        job_pick=job_pick,
    )
    return logfile, jobstart, jobend, jobname, args, build_command(args), options


def main() -> int:
    logfile, jobstart, jobend, jobname, _, cmd, options = parse_args(sys.argv[1:])
    if jobname is not None and jobname not in logfile and jobend > jobstart:
        print(
            f"slurm_run.py: parallel job output must contain {jobname} in log path: {logfile}",
            file=sys.stderr,
        )
        return 1

    fail: Dict[int, int] = {}
    active_jobs: Dict[int, RunningJob] = {}
    numfail = 0

    for job_id in range(jobstart, jobend + 1):
        while len(active_jobs) >= options.max_jobs_run:
            finished_job_id, code = reap_one(active_jobs)
            fail[finished_job_id] = code
            if code != 0:
                numfail += 1

        running_job, immediate_code = launch_job(job_id, jobname, logfile, cmd, options)
        if immediate_code is not None:
            fail[job_id] = immediate_code
            if immediate_code != 0:
                numfail += 1
            continue
        assert running_job is not None
        active_jobs[running_job.process.pid] = running_job

    while active_jobs:
        finished_job_id, code = reap_one(active_jobs)
        fail[finished_job_id] = code
        if code != 0:
            numfail += 1

    failed_jids = sum(1 for code in fail.values() if code != 0)
    if failed_jids:
        njobs = jobend - jobstart + 1
        log_hint = logfile.replace(jobname, "*") if jobname is not None else logfile
        print(f"slurm_run.py: {failed_jids} / {njobs} failed, log is in {log_hint}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        raise
