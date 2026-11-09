"""The gem5 machine that matches QEMU's `virt` machine for QFlex.

What this file is for
---------------------
QFlex takes snapshots of a guest running in QEMU (machine `virt`) and
converts them into gem5 checkpoints (m5.cpt + memory). gem5 can only load a
checkpoint into a machine that has exactly the same objects as the one that
wrote it: it matches each section of m5.cpt ([system.cpu_cluster.cpus0...],
[system.realview.gic], ...) to the simulated object of the same name. This
file builds that machine:
  - qflex_restore.py (called by QPoints/run_gem5.sh) builds it, then loads a
    converted QFlex checkpoint into it;
  - qflex_generate_checkpoint.py builds the same machine to write the
    reference checkpoints from which the conversion template is made.

qflex_generate_checkpoint.py is kept close to its original version, so it
does not import this file: the machine is described in both places and they
MUST stay identical (same CPU count, devices, memory, release, ASID size,
physical address size, counter frequency, workload). Any change here must be
copied there, and the other way round. Differences, all restore-only:
  - the CPU model can be chosen (the atomic CPU is built exactly as in the
    generator);
  - caches are added for timing CPUs, as in starter_fs.py.

The machine
-----------
  - platform QFlex_GICv3 with the memory map of QEMU's device tree: the
    YAML from `dtc -O yaml` is converted by yaml_to_memmap.py (gem5 root);
  - number of cores taken from that device tree;
  - CPU features: ArmAllRelease minus SECURITY and VIRTUALIZATION (QEMU runs
    with secure=off, virtualization=off). The QEMU CPU is lowered to the
    same features with -cpu <model>,id-overrides=<file>, where the file is
    generated from a reference checkpoint of this machine
    (QPoints/cpu_model_features/gen_id_cfg.py);
  - 16-bit ASIDs and 48-bit physical addresses, as configured by the guest
    kernel in QEMU (TCR_EL1.AS = 1, TCR_EL1.IPS = 48 bits);
  - generic timer counter at QEMU's frequency (62.5 MHz);
  - one VirtIO block device on PCI.
"""

import json
import os
import sys

import m5
from m5.objects import *

# gem5's own config helpers: configs/ (for `common`) and configs/example/arm
# (for `devices`) are not on the Python path by default. The gem5 root is
# added too, for yaml_to_memmap.py.
_HERE = os.path.dirname(os.path.abspath(__file__))
_GEM5_ROOT = os.path.join(_HERE, "../../..")
m5.util.addToPath(_GEM5_ROOT)
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

# ---------------------------------------------------------------------------
# Constants of the QEMU `virt` machine
# ---------------------------------------------------------------------------

# Frequency of the generic timer counter (CNTFRQ_EL0) in QEMU 8.1's virt
# machine: 62.5 MHz. The guest kernel read it at boot and converts counter
# ticks to time with it, so gem5's counter must run at the same frequency;
# otherwise time in the restored guest runs too fast or too slow. It is the
# CNTFRQ_EL0 value of register-info.json (0x3b9aca0).
QEMU_CNTFRQ_HZ = 62_500_000

# Physical address size and ASID size the guest kernel configured in QEMU
# (TCR_EL1.IPS = 5 -> 48 bits, TCR_EL1.AS = 1 -> 16-bit ASIDs). gem5's
# defaults are 40 bits and 8-bit ASIDs: with 8-bit ASIDs, two processes
# whose 16-bit ASIDs share the low 8 bits would share TLB entries.
PHYS_ADDR_BITS = 48
LARGE_ASID = True

# Where the DRAM starts in QEMU's virt machine: 1 GiB (0x4000_0000). The DTB
# memory node gives the same value (memmap["dram"]["base"]).
DRAM_BASE = 0x40000000

# The workload addresses below are only used when gem5 itself boots a kernel
# (ArmFsLinux with --kernel). When a QFlex checkpoint is restored, the guest
# kernel, its device tree and its secondary CPUs are already running from
# the QEMU memory image, and gem5 does not write anything at these
# addresses. They are kept identical to qflex_generate_checkpoint.py so the
# two machines stay the same. Values are the usual gem5 ARM layout,
# relative to the start of DRAM:
#   - LOAD_ADDR_OFFSET: added to the kernel's load address, i.e. the kernel
#     image is placed at the start of DRAM;
#   - DTB_OFFSET: gem5 copies its generated device tree 128 MiB into DRAM
#     and passes that address to the kernel;
#   - INITRD_OFFSET_FROM_DTB: an initial ramdisk would go 2 MiB after the
#     DTB (none is used);
#   - CPU_RELEASE_OFFSET_FROM_DTB: "spin-table" address, 8 bytes before the
#     DTB, that gem5's boot code makes secondary CPUs poll before jumping
#     into the kernel (QEMU uses PSCI instead).
LOAD_ADDR_OFFSET = DRAM_BASE
DTB_OFFSET = 0x8000000  # 128 MiB
INITRD_OFFSET_FROM_DTB = 0x200000  # 2 MiB
CPU_RELEASE_OFFSET_FROM_DTB = -8

# Base of the 64 KiB physical range used for memory-mapped m5ops (the
# commands a guest sends to gem5 with the `m5` tool, e.g. `m5 exit` or
# `m5 checkpoint`). Accesses to this range are taken by gem5 instead of
# going to memory or devices.
# TODO: 0x10010000 lies inside QEMU virt's 32-bit PCIe MMIO window
# (0x1000_0000 - 0x3EFE_FFFF), where the guest may place PCI BARs. If a BAR
# of the restored guest overlaps it, its accesses would be taken as m5ops.
# Check against the BARs of the snapshot, or set 0 (disabled) if the guest
# never uses the m5 tool.
M5OPS_BASE = 0x10010000

# SVE vector length, in 128-bit units: QEMU max offers up to 2048 bits
# (the guest kernel reports "maximum available vector length 256 bytes"
# and gives programs 64 bytes by default), so gem5 implements 2048 bits.
SVE_VL_QUADWORDS = 16

# Max SME streaming vector length, in quadwords (128 bits): 16 x 128 = 2048
# bits. QEMU max supports all powers of 2 from 128 to 2048 bits
# (target/arm/tcg/cpu64.c: sme_vq.supported = SVE_VQ_POW2_MAP) and enables
# them all by default. gem5 must offer the same maximum.
SME_VL_QUADWORDS = 16

# Kernel command line, only used when gem5 boots a kernel (see above).
# One option per line:
KERNEL_CMDLINE_OPTIONS = [
    "console=ttyAMA0",  # kernel console on the PL011 UART (serial port)
    "lpj=19988480",  # preset loops_per_jiffy: skips the slow delay-loop
    # calibration at boot, which takes long in simulation
    "norandmaps",  # no address-space randomization of user processes:
    # identical runs give identical addresses
    "rw",  # mount the root file system read-write
]

# CPU models that can be chosen with --cpu, as in starter_fs.py:
# name -> (CPU class, L1 instruction cache, L1 data cache, L2 cache).
# The atomic CPU needs no caches (it accesses memory directly).
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
    """Print an error and stop gem5."""
    print(f"\nERROR: {msg}")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Command-line options
# ---------------------------------------------------------------------------


def add_machine_options(parser):
    """Add the options that define the machine to an argparse parser.

    Used by qflex_restore.py; the same options exist in
    qflex_generate_checkpoint.py.
    """
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


# ---------------------------------------------------------------------------
# Device tree -> memory map
# ---------------------------------------------------------------------------


def load_memmap(yaml_path):
    """Read QEMU's device tree (YAML) and return QFlex_GICv3's memory map.

    The memory map is a dict giving the address, size and interrupt of each
    device of QEMU's virt machine (DRAM, UART, RTC, GIC, PCI, VirtIO...);
    QFlex_GICv3 places its devices at these addresses so that the guest
    kernel finds them where it found them in QEMU. The conversion is done
    by yaml_to_memmap.py, imported as a module (it needs PyYAML in gem5's
    Python). The result is also written to <outdir>/<name>.memmap.json so it
    can be inspected.
    """
    if not os.path.exists(yaml_path):
        fatal(f"YAML file not found: {yaml_path}")
    try:
        import yaml  # PyYAML, used by yaml_to_memmap.py
        import yaml_to_memmap
    except ImportError as e:
        fatal(
            f"cannot import yaml_to_memmap ({e}). It needs PyYAML in the "
            "Python that gem5 uses (e.g. pip install pyyaml)."
        )

    with open(yaml_path) as f:
        dt = yaml.safe_load(f)
    # Keep only the first virtio_mmio node: gem5 has one VirtIO device.
    memmap = yaml_to_memmap.yaml_to_memmap(dt, virtio_indices=[1])

    yaml_base = os.path.splitext(os.path.basename(yaml_path))[0]
    json_path = os.path.join(m5.options.outdir, f"{yaml_base}.memmap.json")
    os.makedirs(m5.options.outdir, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(memmap, f, indent=2)
    print(f"  Memory map: {json_path}")
    return memmap


def _disk_path(args):
    """Path of the raw disk image for the VirtIO block device.

    With --disk-image, look for it as given, then in binaries/ and
    gem5/binaries/. Without it, create an empty 512 MB file in the outdir:
    a restored guest needs a disk device to exist, even if it does not read
    it during the simulated window.
    """
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
        f.seek(512 * 1024 * 1024 - 1)  # sparse 512 MB file
        f.write(b"\0")
    return path


def make_release():
    """CPU features of the machine: ArmAllRelease (every extension gem5
    implements) minus SECURITY (EL3) and VIRTUALIZATION (EL2), which QEMU
    runs without. The id-overrides file given to QEMU must be generated
    from a reference checkpoint built with this same release."""
    release = ArmAllRelease()
    release.remove(ArmExtension("SECURITY"))
    release.remove(ArmExtension("VIRTUALIZATION"))
    return release


# ---------------------------------------------------------------------------
# The machine
# ---------------------------------------------------------------------------


def build_system(args, kernel_path=None, root_device="/dev/vda1"):
    """Build the QFlex machine and its Root. Does not instantiate it.

    kernel_path: only to boot a kernel with gem5 (ArmFsLinux). QFlex
    checkpoints (references and restores) use no kernel: the guest kernel is
    already in the memory image. root_device is then the kernel's root file
    system (first partition of the VirtIO disk).

    Returns (root, info), where info describes what was built.
    """
    # 1. Memory map and core count from QEMU's device tree.
    print("\n[QFlex machine] Device tree")
    memmap = load_memmap(args.yaml)
    cpu_info = memmap.get("cpu", {})
    num_cores = cpu_info.get("num_cores", 2)
    cpu_model = cpu_info.get("cpu_model", "unknown")  # informative only
    if args.num_cores is not None and args.num_cores != num_cores:
        fatal(
            f"--num-cores {args.num_cores} does not match the device tree "
            f"({num_cores} cores in {args.yaml})"
        )
    print(f"  Cores: {num_cores}, QEMU CPU: {cpu_model}")

    # 2. CPU model and memory mode. Timing CPUs (minor, hpi, o3) access
    # memory through caches in "timing" mode; atomic accesses it directly.
    cpu_class, l1i, l1d, l2 = CPU_TYPES[args.cpu]
    mem_mode = cpu_class.memory_mode()
    want_caches = mem_mode == "timing"

    # 3. Workload: how gem5 prepares the machine at start. Without a kernel
    # (QFlex checkpoints), the generic ArmFsWorkload: nothing is loaded,
    # the checkpoint provides the memory. ArmFsLinux would load a kernel and
    # needs its symbols (it stops on __udelay without them).
    workload = ArmFsLinux() if kernel_path else ArmFsWorkload()

    # 4. The system: an ARM system with the QFlex_GICv3 platform, whose
    # devices are placed at QEMU's addresses (memory map from step 1).
    print("[QFlex machine] Platform QFlex_GICv3 (QEMU virt memory map)")
    system = devices.SimpleSystem(
        caches=want_caches,
        mem_size=args.mem_size,
        mem_mode=mem_mode,
        workload=workload,
        platform=QFlex_GICv3(memmap=memmap),
    )

    # Same physical address size and ASID size as the guest kernel uses.
    system.phys_addr_range_64 = PHYS_ADDR_BITS
    system.have_large_asid_64 = LARGE_ASID

    system.sve_vl = SVE_VL_QUADWORDS
    system.sme_vl = SME_VL_QUADWORDS

    # Generic timer counter at QEMU's frequency.
    system.realview.sys_counter.freqs = [QEMU_CNTFRQ_HZ]

    # 5. Main memory (type, size, channels from the options).
    MemConfig.config_mem(args, system)

    # 6. One VirtIO block device on the PCI bus, as in QEMU.
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

    # 7. Connect the buses (memory bus, I/O bus, devices).
    system.connect()

    # 8. The CPUs. The atomic cluster is built exactly as in
    # qflex_generate_checkpoint.py, so references and restores match.
    if args.cpu == "atomic":
        cluster = devices.AtomicCluster(
            system, num_cores, args.cpu_freq, "1.0V"
        )
    else:
        cluster = devices.ArmCpuCluster(
            system, num_cores, args.cpu_freq, "1.0V", cpu_class, l1i, l1d, l2
        )
    system.cpu_cluster = [cluster]

    # 9. CPU features (see make_release).
    system.release = make_release()

    # 10. Caches (only for timing CPUs).
    system.addCaches(
        need_caches=want_caches, last_cache_level=2 if want_caches else 1
    )

    # 11. Workload addresses and m5ops range (see the constants above).
    system.workload.load_addr_offset = LOAD_ADDR_OFFSET
    system.workload.dtb_addr = DRAM_BASE + DTB_OFFSET
    system.workload.initrd_addr = (
        system.workload.dtb_addr + INITRD_OFFSET_FROM_DTB
    )
    system.workload.cpu_release_addr = (
        system.workload.dtb_addr + CPU_RELEASE_OFFSET_FROM_DTB
    )
    system.m5ops_base = M5OPS_BASE

    # Only when gem5 boots a kernel itself (ArmFsLinux): the kernel binary
    # and its command line. Not used for QFlex checkpoints.
    if kernel_path:
        system.workload.object_file = kernel_path
        system.workload.command_line = " ".join(
            KERNEL_CMDLINE_OPTIONS
            + [
                f"root={root_device}",  # root file system device
                f"mem={args.mem_size}",  # memory the kernel may use
            ]
        )

    # 12. gem5's own device tree of this machine, written to the outdir.
    # Only read by a kernel that gem5 boots; a restored guest keeps the
    # device tree it got from QEMU.
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
        disk_path=disk_path,
        dtb=dtb_filename,
        kernel=kernel_path,
    )
    return root, info


def set_max_insts(system, max_insts):
    """Stop the simulation when any CPU thread has executed max_insts
    instructions (option -I)."""
    for cluster in system.cpu_cluster:
        for cpu in cluster.cpus:
            cpu.max_insts_any_thread = max_insts
