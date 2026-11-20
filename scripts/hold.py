#!/usr/bin/env python3
"""Hold CUDA memory without running a compute loop (requires PyTorch)."""

import argparse
import math
import signal
import sys
import threading


def positive_gib(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite positive number")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0,
                        help="CUDA-visible GPU index (default: 0)")
    parser.add_argument("--gib", type=positive_gib, default=60,
                        help="Tensor memory in GiB (default: 60)")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        parser.exit(1, "CUDA is unavailable; use a machine with CUDA-enabled PyTorch.\n")
    if not 0 <= args.gpu < torch.cuda.device_count():
        parser.error("--gpu is outside the CUDA-visible device range")

    stopped = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stopped.set())

    device = torch.device(f"cuda:{args.gpu}")
    buffers = []
    try:
        remaining = int(args.gib * 1024**3)
        while remaining and not stopped.is_set():
            size = min(remaining, 1024**3)
            buffers.append(torch.empty(size, dtype=torch.uint8, device=device))
            remaining -= size
        if not stopped.is_set():
            print(f"Holding {args.gib:g} GiB on {device}. Ctrl+C to release. "
                  "CUDA context overhead adds some extra memory.", flush=True)
            stopped.wait()
    except torch.cuda.OutOfMemoryError:
        print("Insufficient free GPU memory; releasing allocations.", file=sys.stderr)
        return 1
    finally:
        buffers.clear()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
