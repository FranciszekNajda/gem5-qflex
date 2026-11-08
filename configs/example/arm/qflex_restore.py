"""Restore a QFlex checkpoint (converted from a QEMU snapshot) in gem5.

How it fits in the QFlex flow
-----------------------------
  1. QEMU (parallel-qemu) takes a snapshot of the guest and dumps its state:
     register-info.json (CPU registers), the RAM and dev.info.
  2. QPoints' convert.py turns this dump into a gem5 checkpoint directory
     (<snapshot>.gem: m5.cpt + memory files).
  3. QPoints/run_gem5.sh runs this script, which rebuilds the machine and
     loads that checkpoint into it, then lets the guest run.

Options
-------
It accepts the options run_gem5.sh passes to starter_fs.py (-I,
--disk-image, --cpu, --restore, --num-cores, --mem-size, --mem-channels,
--branch-trace), plus --yaml, the QEMU device tree that defines the machine.
The machine is built by qflex_system.py, which describes the same machine as
qflex_generate_checkpoint.py (the reference checkpoints the conversion
template comes from).

Example:
    gem5.opt --outdir=out configs/example/arm/qflex_restore.py \\
        --yaml ../dtb/virt.yaml --restore <snapshot>.gem --cpu atomic -I 100000
"""

import argparse
import os
import sys

import m5
from m5.objects import *

# qflex_system.py is in the same directory as this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qflex_system


def parse_args():
    """Machine options (from qflex_system.py) plus restore/run options."""
    parser = argparse.ArgumentParser(
        description="Restore a QFlex checkpoint in gem5"
    )
    qflex_system.add_machine_options(parser)
    parser.add_argument(
        "--restore",
        type=str,
        required=True,
        help="Checkpoint directory (m5.cpt and *.pmem)",
    )
    parser.add_argument(
        "-I",
        "--max-insts",
        type=int,
        default=None,
        help="Stop when any thread has executed this many instructions",
    )
    parser.add_argument(
        "--sim-ticks",
        type=int,
        default=0,
        help="Stop after this many ticks (default: 0, run until an exit event)",
    )
    parser.add_argument(
        "--checkpoint-after-restore",
        action="store_true",
        help="Write the state gem5 read to <outdir>/reread_checkpoint "
        "(to compare with the converted m5.cpt)",
    )
    parser.add_argument(
        "--branch-trace",
        action="store_true",
        help="Accepted for compatibility with starter_fs.py, which also "
        "parses it without using it",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Build the same machine as the one the checkpoint describes.
    root, info = qflex_system.build_system(args)
    if args.max_insts:
        qflex_system.set_max_insts(root.system, args.max_insts)

    # 2. Create the simulated objects and load the checkpoint into them:
    # gem5 reads each section of m5.cpt into the object of the same name,
    # and the memory files into the main memory.
    print(f"\n[Restore] {args.restore}")
    m5.instantiate(args.restore)
    print(f"  Checkpoint restored at tick {m5.curTick()}")

    # Optional check: write back what gem5 actually read. Comparing this
    # m5.cpt with the converted one shows values gem5 ignored or changed.
    if args.checkpoint_after_restore:
        reread_dir = os.path.join(m5.options.outdir, "reread_checkpoint")
        m5.checkpoint(reread_dir)
        print(f"  State as read by gem5 written to: {reread_dir}/m5.cpt")

    # 3. Run, with the same loop as starter_fs.py: a guest "m5 checkpoint"
    # drops a checkpoint and the run continues; any other exit event
    # (instruction limit, tick limit, guest "m5 exit", end of simulation)
    # ends the run. With --sim-ticks, each m5.simulate() call stops after
    # that many ticks.
    while True:
        if args.sim_ticks > 0:
            event = m5.simulate(args.sim_ticks)
        else:
            event = m5.simulate()
        cause = event.getCause()
        if cause == "checkpoint":
            cpt_dir = os.path.join(m5.options.outdir, f"cpt.{m5.curTick()}")
            m5.checkpoint(cpt_dir)
            print(f"Checkpoint dropped at tick {m5.curTick()}: {cpt_dir}")
            if args.sim_ticks > 0:
                break
            continue
        print(f"Exit at tick {m5.curTick()}: {cause} ({event.getCode()})")
        break


# gem5 runs the configuration script with __name__ == "__m5_main__".
if __name__ == "__m5_main__":
    main()
    