"""Restore a QFlex checkpoint (converted from a QEMU snapshot) in gem5.

Called by QPoints/run_gem5.sh. It accepts the options run_gem5.sh passes to
starter_fs.py (-I, --disk-image, --cpu, --restore, --num-cores, --mem-size,
--mem-channels, --branch-trace), plus --yaml, the QEMU device tree that
defines the machine.

The machine is built by qflex_system.py, the same code that produced the
reference checkpoints the conversion template comes from.

Example:
    gem5.opt --outdir=out configs/example/arm/qflex_restore.py \\
        --yaml ../dtb/virt.yaml --restore <snapshot>.gem --cpu atomic -I 100000
"""

import argparse
import os
import sys

import m5
from m5.objects import *

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qflex_system


def parse_args():
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

    root, info = qflex_system.build_system(args)
    if args.max_insts:
        qflex_system.set_max_insts(root.system, args.max_insts)

    print(f"\n[Restore] {args.restore}")
    m5.instantiate(args.restore)
    print(f"  Checkpoint restored at tick {m5.curTick()}")

    if args.checkpoint_after_restore:
        reread_dir = os.path.join(m5.options.outdir, "reread_checkpoint")
        m5.checkpoint(reread_dir)
        print(f"  State as read by gem5 written to: {reread_dir}/m5.cpt")

    # Same loop as starter_fs.py: a guest "m5 checkpoint" drops a checkpoint,
    # any other exit event ends the run.
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


if __name__ == "__m5_main__":
    main()
