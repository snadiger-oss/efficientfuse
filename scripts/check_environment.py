"""Read-only check of CUDA, software versions, UUID mapping and NVML energy."""

import argparse
import json
from pathlib import Path

from greenfuse.measurement.environment import environment_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--exclusive-gpu", action="store_true", help="Assert that this job owns the whole physical GPU")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = environment_report(args.device, exclusive_gpu=args.exclusive_gpu)
    text = json.dumps(report, indent=2, allow_nan=False) + "\n"
    print(text, end="")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    return 0 if report["status"] != "unavailable" else 1


if __name__ == "__main__":
    raise SystemExit(main())
