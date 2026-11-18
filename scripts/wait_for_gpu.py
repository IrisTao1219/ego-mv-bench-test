#!/usr/bin/env python3
"""Wait for one NVIDIA GPU to stay idle, then exec a command once."""

import argparse
import csv
import io
import math
import os
import subprocess
import sys
import time


def nonnegative(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("must be a finite nonnegative number")
    return number


def query(gpu, fields, kind="gpu"):
    result = subprocess.run(
        ["nvidia-smi", "-i", gpu, f"--query-{kind}={fields}",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=15,
    )
    return list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True, help="Physical GPU index or UUID")
    parser.add_argument("--max-used-mib", type=nonnegative, default=500)
    parser.add_argument("--max-util", type=nonnegative, default=5,
                        help="Maximum GPU utilization percentage (default: 5)")
    parser.add_argument("--min-free-mib", type=nonnegative, default=71680,
                        help="Minimum free memory in MiB (default: 71680 = 70 GiB)")
    parser.add_argument("--interval", type=nonnegative, default=60,
                        help="Polling interval in seconds (default: 60)")
    parser.add_argument("--stable-seconds", type=nonnegative, default=0,
                        help="Required continuous idle seconds (default: 0, start immediately)")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="Command after -- (do not set CUDA_VISIBLE_DEVICES again)")
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command or args.interval <= 0 or args.max_util > 100:
        parser.error("supply a command after --, interval > 0, and max-util <= 100")
    if "," in args.gpu:
        parser.error("--gpu must select exactly one GPU")

    # Resolve once to a UUID so CUDA selects the same card as nvidia-smi.
    try:
        rows = query(args.gpu, "uuid")
        if len(rows) != 1 or len(rows[0]) != 1 or not rows[0][0].startswith("GPU-"):
            raise ValueError("expected exactly one full GPU UUID")
        gpu = rows[0][0].strip()
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        parser.exit(1, f"Cannot select GPU {args.gpu}: {exc}\n")

    idle_since = None
    while True:
        try:
            rows = query(gpu, "memory.used,memory.free,utilization.gpu")
            if len(rows) != 1 or len(rows[0]) != 3:
                raise ValueError("unexpected GPU status")
            used, free, util = map(float, rows[0])
            if not all(math.isfinite(n) for n in (used, free, util)):
                raise ValueError("unavailable GPU metrics")
            processes = query(gpu, "pid", "compute-apps")
            idle = (not processes and used <= args.max_used_mib
                    and free >= args.min_free_mib and util <= args.max_util)
            now = time.monotonic()
            idle_since = (now if idle_since is None else idle_since) if idle else None
            elapsed = 0 if idle_since is None else now - idle_since
            print(f"GPU {args.gpu}: used={used:.0f} MiB, free={free:.0f} MiB, "
                  f"util={util:.0f}%, processes={len(processes)}, "
                  f"idle={elapsed:.0f}/{args.stable_seconds:g}s", flush=True)
            if idle and elapsed >= args.stable_seconds:
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu
                print(f"Starting with CUDA_VISIBLE_DEVICES={gpu}: {command!r}", flush=True)
                os.execvp(command[0], command)
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            idle_since = None
            print(f"GPU check failed; continuing to wait: {exc}", file=sys.stderr, flush=True)
            # A failed launch should exit, not start another monitoring cycle.
            if isinstance(exc, OSError) and exc.filename == command[0]:
                return 1
        time.sleep(args.interval)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
