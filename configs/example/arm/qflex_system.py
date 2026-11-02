"""The gem5 machine that matches QEMU's `virt` machine for QFlex.

Used by qflex_restore.py (called by run_gem5.sh) to rebuild the machine
before loading a converted QFlex checkpoint.

The code is taken from qflex_generate_checkpoint.py (steps 1-10), which
produces the reference checkpoints the conversion template comes from. That
script is not imported or run from here, so the machine is described in both
places and they MUST stay identical: gem5 restores a checkpoint by matching
each section of m5.cpt to the object of the same name, so any difference
(CPU count, devices, memory, workload) breaks the restore. Any change to the
machine in qflex_generate_checkpoint.py must be copied here, and the other way
round.

Differences with qflex_generate_checkpoint.py, all restore-only:
  - CPU model selectable (atomic is built exactly as in the generator);
  - caches added for timing CPUs, as in starter_fs.py.

The machine:
  - platform QFlex_GICv3, with the memory map of QEMU's device tree
    (YAML from `dtc -O yaml`, converted by yaml_to_memmap.py);
  - number of cores and ARM release taken from that device tree;
  - ARM release without SECURITY and VIRTUALIZATION (QEMU runs with
    secure=off, virtualization=off);
  - one VirtIO block device;
  - CPU model selectable (same table as starter_fs.py). Caches are only
    added for timing CPUs, as in starter_fs.py.
"""

import json
import os
import shutil
import subprocess
import sys

import m5
from m5.objects import *

_HERE = os.path.dirname(os.path.abspath(__file__))
m5.util.addToPath(os.path.join(_HERE, "../../.."))
m5.util.addToPath(os.path.join(_HERE, "../.."))

import devices
from common import (
    MemConfig,
    ObjectList,
)
from common.cores.arm import (
    HPI,
    O3_ARM_v7a,
)

YAML_TO_MEMMAP = os.path.join(_HERE, "../../../yaml_to_memmap.py")

# Same table as starter_fs.py: (cpu_class, l1i, l1d, l2).
CPU_TYPES = {
    "atomic": (AtomicSimpleCPU, None, None, None),
    "minor": (MinorCPU, devices.L1I, devices.L1D, devices.L2),
    "hpi": (HPI.HPI, HPI.HPI_ICache, HPI.HPI_DCache, HPI.HPI_L2),
    "o3": (
        O3_ARM_v7a.O3_ARM_v7a_3,
        O3_ARM_v7a.O3_ARM_v7a_ICache,
        O3_ARM_v7a.O3_ARM_v7a_DCache,
        O3_ARM_v7a.O3_ARM_v7aL2,
    ),
}


def fatal(msg):
    print(f"\nERROR: {msg}")
    sys.exit(1)


def add_machine_options(parser):
    """Options that define the machine. Shared by generation and restore."""
    parser.add_argument(
        "--yaml",
        type=str,
        required=True,
        help="QEMU device tree as YAML (dtc -O yaml), e.g. dtb/virt.yaml",
    )
    parser.add_argument(
        "--cpu",
        type=str,
        choices=list(CPU_TYPES.keys()),
        default="atomic",
        help="CPU model (default: atomic)",
    )
    parser.add_argument(
        "--cpu-freq",
        type=str,
        default="4GHz",
        help="CPU frequency (default: 4GHz)",
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        default=None,
        help="Number of cores. The device tree decides; if given, it must "
        "match the device tree (a mismatch is an error).",
    )
    parser.add_argument(
        "--mem-type",
        default="DDR3_1600_8x8",
        choices=ObjectList.mem_list.get_names(),
        help="Memory type (default: DDR3_1600_8x8)",
    )
    parser.add_argument(
        "--mem-size",
        type=str,
        default="8GiB",
        help="Physical memory size (default: 8GiB)",
    )
    parser.add_argument(
        "--mem-channels",
        type=int,
        default=1,
        help="Number of memory channels (default: 1)",
    )
    parser.add_argument(
        "--mem-ranks",
        type=int,
        default=None,
        help="Number of memory ranks per channel",
    )
    parser.add_argument(
        "--disk-image",
        type=str,
        default=None,
        help="Raw disk image (default: an empty 512 MB disk in the outdir)",
    )


def load_memmap(yaml_path):
    """Convert the device tree YAML to the memory-map dict of QFlex_GICv3."""
    if not os.path.exists(yaml_path):
        fatal(f"YAML file not found: {yaml_path}")
    if not os.path.exists(YAML_TO_MEMMAP):
        fatal(f"yaml_to_memmap.py not found at: {YAML_TO_MEMMAP}")

    # Under gem5, sys.executable is gem5.opt: find a real Python interpreter.
    python_exe = (
        os.environ.get("PYTHON3")
        or shutil.which("python3")
        or shutil.which("python")
    )
    if not python_exe:
        fatal("Could not find a Python interpreter (python3/python)")

    yaml_base = os.path.splitext(os.path.basename(yaml_path))[0]
    json_path = os.path.abspath(f"{yaml_base}.memmap.json")

    try:
        result = subprocess.run(
            [
                python_exe,
                YAML_TO_MEMMAP,
                yaml_path,
                "-o",
                json_path,
                "--virtio-devices",
                "1",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        fatal(f"Failed to convert YAML to JSON:\n{e.stderr}")

    # Older helper versions print the JSON instead of writing it.
    if not os.path.exists(json_path):
        out = (result.stdout or "").strip()
        start = out.find("{")
        if start != -1:
            try:
                with open(json_path, "w") as f:
                    json.dump(json.loads(out[start:]), f, indent=2)
            except Exception:
                pass
    if not os.path.exists(json_path):
        fatal(f"Memmap JSON was not created: {json_path}")

    with open(json_path) as f:
        memmap = json.load(f)
    print(f"  Memmap JSON: {json_path}")
    return memmap


def _disk_path(args):
    if args.disk_image:
        for path in (
            args.disk_image,
            f"binaries/{args.disk_image}",
            f"gem5/binaries/{args.disk_image}",
        ):
            if os.path.exists(path):
                return path
        fatal(f"Disk image not found: {args.disk_image}")

    path = os.path.join(m5.options.outdir, "qflex_disk.img")
    os.makedirs(m5.options.outdir, exist_ok=True)
    with open(path, "wb") as f:
        f.seek(512 * 1024 * 1024 - 1)  # empty 512 MB disk
        f.write(b"\0")
    return path


def build_system(args, kernel_path=None, root_device="/dev/vda1"):
    """Build the QFlex machine and its Root. Does not instantiate it.

    kernel_path: only for booting a kernel. QFlex checkpoints (references
    and restores) use no kernel: the guest kernel is already in memory.

    Returns (root, info) where info describes what was built.
    """
    print("\n[QFlex machine] Device tree")
    memmap = load_memmap(args.yaml)
    cpu_info = memmap.get("cpu", {})
    num_cores = cpu_info.get("num_cores", 2)
    cpu_model = cpu_info.get("cpu_model", "cortex-a57")
    arm_release = cpu_info.get("release", "armv8")
    if args.num_cores is not None and args.num_cores != num_cores:
        fatal(
            f"--num-cores {args.num_cores} does not match the device tree "
            f"({num_cores} cores in {args.yaml})"
        )
    print(f"  Cores: {num_cores}, model: {cpu_model}, release: {arm_release}")

    cpu_class, l1i, l1d, l2 = CPU_TYPES[args.cpu]
    mem_mode = cpu_class.memory_mode()
    want_caches = mem_mode == "timing"

    # ArmFsLinux needs the kernel's symbols (it panics on __udelay without
    # them); without a kernel, use the generic full-system workload.
    workload = ArmFsLinux() if kernel_path else ArmFsWorkload()

    print("[QFlex machine] Platform QFlex_GICv3 (QEMU virt memory map)")
    system = devices.SimpleSystem(
        caches=want_caches,
        mem_size=args.mem_size,
        mem_mode=mem_mode,
        workload=workload,
        platform=QFlex_GICv3(memmap=memmap),
    )

    MemConfig.config_mem(args, system)

    disk_path = _disk_path(args)
    system.pci_devices = [
        PciVirtIO(
            vio=VirtIOBlock(
                image=RawDiskImage(read_only=False, image_file=disk_path)
            )
        )
    ]
    for dev in system.pci_devices:
        system.attach_pci(dev)

    system.connect()

    # The atomic cluster is kept exactly as before so that reference
    # checkpoints do not change.
    if args.cpu == "atomic":
        cluster = devices.AtomicCluster(
            system, num_cores, args.cpu_freq, "1.0V"
        )
    else:
        cluster = devices.ArmCpuCluster(
            system, num_cores, args.cpu_freq, "1.0V", cpu_class, l1i, l1d, l2
        )
    system.cpu_cluster = [cluster]

    release = Armv82() if arm_release == "armv8.2" else Armv8()
    release.remove(ArmExtension("SECURITY"))
    release.remove(ArmExtension("VIRTUALIZATION"))
    system.release = release

    system.addCaches(
        need_caches=want_caches, last_cache_level=2 if want_caches else 1
    )

    system.workload.load_addr_offset = 0x40000000
    system.workload.dtb_addr = 0x40000000 + 0x8000000
    system.workload.initrd_addr = system.workload.dtb_addr + 0x200000
    system.workload.cpu_release_addr = system.workload.dtb_addr - 8
    system.m5ops_base = 0x10010000

    if kernel_path:
        system.workload.object_file = kernel_path
        system.workload.command_line = (
            "console=ttyAMA0 lpj=19988480 norandmaps "
            f"root={root_device} rw mem={args.mem_size}"
        )

    dtb_filename = os.path.join(m5.options.outdir, "qflex_system.dtb")
    system.workload.dtb_filename = dtb_filename

    root = Root(full_system=True, system=system)
    system.generateDtb(dtb_filename)

    print(
        f"[QFlex machine] {num_cores} x {args.cpu} @ {args.cpu_freq}, "
        f"caches: {'yes' if want_caches else 'no'}, "
        f"memory: {args.mem_size} {args.mem_type}, disk: {disk_path}"
    )
    info = dict(
        num_cores=num_cores,
        cpu_model=cpu_model,
        arm_release=arm_release,
        disk_path=disk_path,
        dtb=dtb_filename,
        kernel=kernel_path,
    )
    return root, info


def set_max_insts(system, max_insts):
    """Stop when any thread has executed max_insts instructions (-I)."""
    for cluster in system.cpu_cluster:
        for cpu in cluster.cpus:
            cpu.max_insts_any_thread = max_insts
