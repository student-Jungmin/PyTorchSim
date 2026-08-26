"""The gem5 half of the TOGSim trace: per-tile cycles into a cycle table.

The producer itself is the compiler's, C++ and .so both.
"""
import os


VECTOR_COMPUTE = 0
MATMUL_COMPUTE = 1
MATMUL_PRELOAD = 2


def overlapping_cycle(cycle, compute_type, x_offset, w_offset):
    """The pipeline-overlapped portion of cycle, by compute type."""
    if compute_type <= VECTOR_COMPUTE:
        return 0
    offset = w_offset if compute_type == MATMUL_PRELOAD else x_offset
    return max(int(cycle) - int(offset), 0)


def cycle_table(compute_types, cycle_list, x_offset, w_offset):
    """[(cycle, overlapping_cycle), ...] indexed by tile_id.

    compute_types comes from the compiler, cycle_list from gem5; both are in
    tile_id order.
    """
    if len(cycle_list) != len(compute_types):
        raise ValueError(
            f"gem5 returned {len(cycle_list)} cycle sample(s) for "
            f"{len(compute_types)} compute tile(s): a marker fired a different "
            f"number of times than there are tiles, so the table would be keyed "
            f"by the wrong samples")
    return [(int(c), overlapping_cycle(c, t, x_offset, w_offset))
            for c, t in zip(cycle_list, compute_types)]


def dump_cycle_table_tsv(table, path, origins=None):
    """cycle<TAB>overlapping per line, in tile_id order, for TOGSim's loader.

    origins is appended as a trailing comment line, which the loader stops at.
    """
    with open(path, "w") as fh:
        for cycle, overlapping in table:
            fh.write("%d\t%d\n" % (int(cycle), int(overlapping)))
        if origins:
            fh.write("# origins: %s\n" % ", ".join(sorted(str(o) for o in origins)))
    return path


def default_include_dir():
    """Where togsim_runtime.h lives. provenance hashes it into the cache key."""
    root = os.environ.get("TORCHSIM_DIR")
    if not root:
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
    return os.path.join(root, "TOGSim", "include")
