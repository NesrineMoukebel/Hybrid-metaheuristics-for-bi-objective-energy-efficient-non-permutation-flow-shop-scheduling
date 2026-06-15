import os
import math

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

# Instances to generate TOU files for
INSTANCES = [
    (300, 40, 6),
    (200, 20, 4),
    (50,  15, 9),
    (20,   5, 7),
]

# Lambda values to test (lambda=5 already exists, generate 4, 6, 7)
LAMBDA_VALUES = [4, 6, 7]

# TOU tariff scheme (fixed, same as original)
PRICES = [0.08, 0.12, 0.08, 0.12, 0.08, 0.04]

# Period duration ratios (same as original: H/12, H/6, H/4, H/6, H/12, H/4)
DURATION_RATIOS = [1/12, 1/6, 1/4, 1/6, 1/12, 1/4]

# Path to your processing time files
INPUT_DIR  = "../"   # folder containing VFR*_Gap.txt files
OUTPUT_DIR = "./"   # folder to write new TOU files into

# ─────────────────────────────────────────────
# STEP 1: READ PROCESSING TIMES
# ─────────────────────────────────────────────

def read_processing_times(filepath):
    """
    Read a VFR processing time file.
    Returns (nb_jobs, nb_machines, processing_matrix).
    processing_matrix[i][j] = processing time of job i on machine j.
    """
    with open(filepath, 'r') as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]

    # First line: nb_jobs nb_machines
    first = lines[0].split()
    nb_jobs     = int(first[0])
    nb_machines = int(first[1])

    processing_matrix = []
    for line in lines[1:]:
        tokens = line.split()
        # Format: 0 p0 1 p1 2 p2 ... → take every second value starting at index 1
        times = [int(tokens[2*j + 1]) for j in range(nb_machines)]
        processing_matrix.append(times)

    return nb_jobs, nb_machines, processing_matrix


# ─────────────────────────────────────────────
# STEP 2: COMPUTE HORIZON
# ─────────────────────────────────────────────

def compute_horizon(processing_matrix, nb_machines, lam):
    """
    H = lambda * (sum of all p_ij) / M
    Rounded to nearest integer.
    """
    total = sum(
        processing_matrix[i][j]
        for i in range(len(processing_matrix))
        for j in range(nb_machines)
    )
    H = lam * total / nb_machines
    return int(round(H))


# ─────────────────────────────────────────────
# STEP 3: COMPUTE PERIOD STARTS AND ENDS
# ─────────────────────────────────────────────

def compute_periods(H, duration_ratios):
    """
    Given horizon H and duration ratios, compute:
    - period_durations: list of int durations
    - starts: list of period start times (1-indexed, like original file)
    - ends:   list of period end times

    Original file format:
      starts: 1  d1+1  d1+d2+1  ...
      ends:   d1  d1+d2  ...  H
    """
    # Compute raw durations (float first, then round)
    raw_durations = [H * r for r in duration_ratios]

    # Round to integers — adjust last period so they sum exactly to H
    durations = [int(round(d)) for d in raw_durations]

    # Fix rounding error on last period
    diff = H - sum(durations)
    durations[-1] += diff

    # Compute starts and ends (1-based, matching original file format)
    starts = []
    ends   = []
    cumulative = 0
    for d in durations:
        starts.append(cumulative + 1)
        ends.append(cumulative + d)
        cumulative += d

    return durations, starts, ends


# ─────────────────────────────────────────────
# STEP 4: WRITE TOU FILE
# ─────────────────────────────────────────────

def write_tou_file(filepath, H, starts, ends, prices):
    """
    Write TOU file in the same format as the original:
      Line 1: H (horizon)
      Line 2: starts separated by spaces
      Line 3: ends separated by spaces
      Line 4: prices separated by spaces
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True) if os.path.dirname(filepath) else None

    with open(filepath, 'w') as f:
        f.write(f"{H}\n")
        f.write(" ".join(str(s) for s in starts) + "\n")
        f.write(" ".join(str(e) for e in ends)   + "\n")
        f.write(" ".join(str(p) for p in prices) + "\n")

    print(f"  Written: {filepath}")


# ─────────────────────────────────────────────
# STEP 5: NAMING CONVENTION
# ─────────────────────────────────────────────

def tou_filename(nb_jobs, nb_machines, instance, lam, profile="6CW"):
    """
    Naming convention:
      VFR{jobs}_{machines}_{instance}_Gap__6CW_L{lambda}.txt

    Examples:
      VFR300_40_6_Gap__6CW_L4.txt
      VFR300_40_6_Gap__6CW_L6.txt
      VFR300_40_6_Gap__6CW_L7.txt

    The original lambda=5 file is:
      VFR300_40_6_Gap__6CW.txt   (no lambda suffix = lambda 5)
    """
    return f"VFR{nb_jobs}_{nb_machines}_{instance}_Gap__{profile}_L{lam}.txt"


def proc_filename(nb_jobs, nb_machines, instance):
    return f"VFR{nb_jobs}_{nb_machines}_{instance}_Gap.txt"


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("="*60)
    print("TOU File Generator — Lambda Sensitivity Analysis")
    print("="*60)

    for (nb_jobs, nb_machines, instance) in INSTANCES:
        proc_file = os.path.join(INPUT_DIR, proc_filename(nb_jobs, nb_machines, instance))

        print(f"\nInstance: {nb_jobs} jobs, {nb_machines} machines, instance {instance}")
        print(f"  Reading: {proc_file}")

        if not os.path.exists(proc_file):
            print(f"  ERROR: File not found — {proc_file}")
            continue

        _, nb_m, processing_matrix = read_processing_times(proc_file)

        # Sanity check
        assert nb_m == nb_machines, \
            f"Machine count mismatch: file says {nb_m}, expected {nb_machines}"

        for lam in LAMBDA_VALUES:
            H = compute_horizon(processing_matrix, nb_machines, lam)
            durations, starts, ends = compute_periods(H, DURATION_RATIOS)

            out_name = tou_filename(nb_jobs, nb_machines, instance, lam)
            out_path = os.path.join(OUTPUT_DIR, out_name)

            write_tou_file(out_path, H, starts, ends, PRICES)

            # Print summary for verification
            print(f"    λ={lam} → H={H} | durations={durations} | sum={sum(durations)}")

    print("\n" + "="*60)
    print("Done. Verify that sum(durations) == H for each file.")
    print("="*60)


if __name__ == "__main__":
    main()