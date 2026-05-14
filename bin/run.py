#!/usr/bin/env python3
import re
import shlex
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


USAGE = "usage: run.py log-file command-line arguments..."


@dataclass
class RunningJob:
    job_id: int
    log_path: str
    process: subprocess.Popen


def shell_quote(arg: str) -> str:
    return shlex.quote(arg)


def build_command(args: List[str]) -> str:
    return " ".join(shell_quote(arg) for arg in args) + " "


def detect_max_jobs(using_gpu: bool, num_jobs: int) -> int:
    max_jobs = 0
    if using_gpu:
      try:
          proc = subprocess.run(
              ["nvidia-smi", "-L"],
              check=False,
              capture_output=True,
              text=True,
          )
          if proc.returncode == 0:
              max_jobs = sum(1 for line in proc.stdout.splitlines() if line.strip())
      except OSError:
          max_jobs = 0

      if max_jobs == 0:
          max_jobs = 1
          print(
              f"run.py: Warning: failed to detect number of GPUs from nvidia-smi, using {max_jobs}",
              file=sys.stderr,
          )
      return max_jobs

    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        with cpuinfo.open() as handle:
            max_jobs = sum(1 for line in handle if line.startswith("processor"))
        if max_jobs == 0:
            print(
                "run.py: Warning: failed to detect any processors from /proc/cpuinfo",
                file=sys.stderr,
            )
            max_jobs = 10
    else:
        try:
            proc = subprocess.run(
                ["sysctl", "-a"],
                check=False,
                capture_output=True,
                text=True,
            )
            if proc.returncode == 0:
                for line in proc.stdout.splitlines():
                    match = re.search(r"hw\.ncpu\s*[:=]\s*(\d+)", line)
                    if match:
                        max_jobs = int(match.group(1))
                        break
        except OSError:
            max_jobs = 0
        if max_jobs == 0:
            print(
                "run.py: Warning: failed to detect any processors from sysctl -a",
                file=sys.stderr,
            )
            max_jobs = 10

    if num_jobs > max_jobs and num_jobs < 1.4 * max_jobs:
        max_jobs = num_jobs
    return max_jobs if max_jobs > 0 else 32


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
    if re.search(r"\S", log_line):
        return "run"
    return "run"


def ensure_parent_dir(logfile: str) -> None:
    parent = Path(logfile).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def format_now() -> str:
    return time.strftime("%a %b %d %H:%M:%S %Z %Y", time.localtime())


def write_log_header(logfile: str, cmd: str) -> int:
    ensure_parent_dir(logfile)
    start_time = int(time.time())
    with open(logfile, "w", encoding="utf-8") as handle:
        handle.write(f"# {cmd}\n")
        handle.write(f"# Started at {format_now()}\n")
        handle.write("#\n")
    return start_time


def append_log_footer(logfile: str, returncode: int, start_time: int) -> None:
    end_time = int(time.time())
    if returncode < 0:
        sig = -returncode
        return_str = f"code 0; signal {sig}"
    else:
        return_str = f"code {returncode}"

    with open(logfile, "a", encoding="utf-8") as handle:
        handle.write(f"# Accounting: time={end_time - start_time} threads=1\n")
        handle.write(
            f"# Ended ({return_str}) at {format_now()}, elapsed time {end_time - start_time} seconds\n"
        )


def launch_job(job_id: int, jobname: Optional[str], logfile: str, cmd: str, job_pick: str) -> Tuple[Optional[RunningJob], Optional[int]]:
    if jobname is not None:
        cmd = cmd.replace(jobname, str(job_id))
        logfile = logfile.replace(jobname, str(job_id))

    pick_action = inspect_log_for_pick(logfile, job_pick)
    if pick_action == "skip_success":
        return None, 0
    if pick_action == "skip_failure":
        return None, 1

    start_time = write_log_header(logfile, cmd)
    log_handle = open(logfile, "a", encoding="utf-8")
    process = subprocess.Popen(
        ["bash", "-c", f"( {cmd})"],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    setattr(process, "_runpy_log_handle", log_handle)
    setattr(process, "_runpy_start_time", start_time)
    return RunningJob(job_id=job_id, log_path=logfile, process=process), None


def finalize_job(running_job: RunningJob) -> int:
    process = running_job.process
    returncode = process.wait()
    log_handle = getattr(process, "_runpy_log_handle")
    log_handle.close()
    start_time = getattr(process, "_runpy_start_time")
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
        time.sleep(0.1)


def parse_args(argv: List[str]) -> Tuple[str, int, int, int, str, Optional[str], List[str], str]:
    if len(argv) < 2:
        raise SystemExit(USAGE)

    job_pick = "all"
    max_jobs_run = -1
    jobstart = 1
    jobend = 1
    jobname = None
    using_gpu = False
    args = list(argv)

    for _ in range(2):
        while len(args) >= 2 and args[0].startswith("-"):
            switch = args.pop(0)
            if switch == "-V":
                continue
            if switch in {"--max-jobs-run", "-tc"}:
                if not args:
                    raise SystemExit(f"run.py: invalid option {switch}")
                new_constraint = int(args.pop(0))
                if max_jobs_run <= 0:
                    max_jobs_run = new_constraint
                else:
                    max_jobs_run = min(max_jobs_run, new_constraint)
                if max_jobs_run <= 0:
                    raise SystemExit(f"run.py: invalid option --max-jobs-run {max_jobs_run}")
                continue

            if not args:
                raise SystemExit(f"run.py: missing argument to {switch}")
            argument = args.pop(0)
            if argument.startswith("--"):
                print(
                    f"run.py: WARNING: suspicious argument '{argument}' to {switch}; starts with '-'",
                    file=sys.stderr,
                )
            if switch == "-sync" and re.match(r"^[yY]", argument):
                continue
            if switch == "-pe":
                if not args:
                    raise SystemExit("run.py: missing second argument to -pe")
                args.pop(0)
                continue
            if switch == "--gpu":
                using_gpu = argument not in {"", "0"}
                continue
            if switch == "--pick":
                if argument not in {"all", "failed", "incomplete"}:
                    raise SystemExit(
                        "run.py: ERROR: --pick argument must be one of 'all', 'failed' or 'incomplete'"
                    )
                job_pick = argument
                continue

        if args and re.match(r"^([\w_][\w\d_]*)=(\d+):(\d+)$", args[0]):
            match = re.match(r"^([\w_][\w\d_]*)=(\d+):(\d+)$", args.pop(0))
            assert match is not None
            jobname = match.group(1)
            jobstart = int(match.group(2))
            jobend = int(match.group(3))
            if jobstart > jobend:
                raise SystemExit(f"run.py: invalid job range {jobname}={jobstart}:{jobend}")
            if jobstart <= 0:
                raise SystemExit(
                    f"run.py: invalid job range {jobname}={jobstart}:{jobend}, start must be strictly positive (this is required for GridEngine compatibility)."
                )
        elif args and re.match(r"^([\w_][\w\d_]*)=(\d+)$", args[0]):
            match = re.match(r"^([\w_][\w\d_]*)=(\d+)$", args.pop(0))
            assert match is not None
            jobname = match.group(1)
            jobstart = int(match.group(2))
            jobend = int(match.group(2))
        elif args and re.match(r".+=.*:.*$", args[0]):
            print(f"run.py: Warning: suspicious first argument to run.py: {args[0]}", file=sys.stderr)

    logfile = args.pop(0)
    if not args:
        raise SystemExit(USAGE)

    num_jobs = jobend - jobstart + 1
    if max_jobs_run == -1:
        max_jobs_run = detect_max_jobs(using_gpu, num_jobs)

    return logfile, jobstart, jobend, max_jobs_run, job_pick, jobname, args, build_command(args)


def main() -> int:
    logfile, jobstart, jobend, max_jobs_run, job_pick, jobname, _, cmd = parse_args(sys.argv[1:])

    if jobname is not None and jobname not in logfile and jobend > jobstart:
        print(
            f"run.py: you are trying to run a parallel job but you are putting the output into just one log file ({logfile})",
            file=sys.stderr,
        )
        return 1

    fail: Dict[int, int] = {}
    active_jobs: Dict[int, RunningJob] = {}
    numfail = 0

    for job_id in range(jobstart, jobend + 1):
        while len(active_jobs) >= max_jobs_run:
            finished_job_id, code = reap_one(active_jobs)
            fail[finished_job_id] = code
            if code != 0:
                numfail += 1

        running_job, immediate_code = launch_job(job_id, jobname, logfile, cmd, job_pick)
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

    failed_jids = 0
    for job_id in range(jobstart, jobend + 1):
        if job_id not in fail:
            raise RuntimeError(
                "run.py: Sanity check failed: we have indication that some jobs are running even after we waited for all jobs to finish"
            )
        if fail[job_id] != 0:
            failed_jids += 1

    if failed_jids != numfail:
        raise RuntimeError(
            f"run.py: Sanity check failed: cannot find out how many jobs failed ({failed_jids} x {numfail})."
        )

    ret = 1 if numfail > 0 else 0
    if ret != 0:
        njobs = jobend - jobstart + 1
        if njobs == 1:
            if jobname is not None:
                logfile = logfile.replace(jobname, str(jobstart))
            print(f"run.py: job failed, log is in {logfile}", file=sys.stderr)
            if "JOB" in logfile:
                print(
                    "run.py: probably you forgot to put JOB=1:$nj in your script.",
                    file=sys.stderr,
                )
        else:
            if jobname is not None:
                logfile = logfile.replace(jobname, "*")
            print(f"run.py: {numfail} / {njobs} failed, log is in {logfile}", file=sys.stderr)

    return ret


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        raise
