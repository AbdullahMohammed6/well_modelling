import os
import glob
import numpy as np
import pandas as pd
import roxar

"""
Import all possible well trajectory/path text files under mainpath into RMS.

Supported input formats per file:
- 4+ columns: interpreted as X, Y, Z(TVD), MD using first 4 columns.
- 3 columns: interpreted as X, Y, Z(TVD), with MD generated from cumulative distance.

The script scans recursively under `mainpath` for .txt files, including paths such as:
- /main/*.txt
- /main/realization-*/**/COOK.txt
"""

# -------------------------------------------------
# Configuration
# -------------------------------------------------
mainpath = r"/main/"   # <-- Change to your folder


# -------------------------------------------------
# Helpers
# -------------------------------------------------
def discover_trajectory_files(base_folder: str) -> list[str]:
    """Find all candidate trajectory files recursively under base_folder."""
    direct_pattern = os.path.join(base_folder, "*.txt")
    recursive_pattern = os.path.join(base_folder, "**", "*.txt")
    cook_pattern = os.path.join(base_folder, "realization-*", "**", "COOK.txt")

    files = set(glob.glob(direct_pattern))
    files.update(glob.glob(recursive_pattern, recursive=True))
    files.update(glob.glob(cook_pattern, recursive=True))

    return sorted(f for f in files if os.path.isfile(f))


def _read_numeric_text_file(filepath: str) -> np.ndarray:
    """Read numeric txt with flexible separators, headers, and comments."""
    # Try pandas first for mixed spacing/tabs and optional header lines.
    df = pd.read_csv(
        filepath,
        sep=r"[\s,;]+",
        engine="python",
        comment="#",
        header=None,
    )

    # Coerce to numeric and drop non-numeric rows.
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if df.empty:
        raise ValueError("No numeric trajectory rows found")

    arr = df.to_numpy(dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def load_trajectory(filepath: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return x, y, z, md arrays from a trajectory file."""
    arr = _read_numeric_text_file(filepath)

    if arr.shape[1] >= 4:
        x = arr[:, 0]
        y = arr[:, 1]
        z = arr[:, 2]
        md = arr[:, 3]
    elif arr.shape[1] == 3:
        x = arr[:, 0]
        y = arr[:, 1]
        z = arr[:, 2]
        xyz = np.column_stack((x, y, z))
        if len(xyz) == 1:
            md = np.array([0.0])
        else:
            seg = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
            md = np.concatenate(([0.0], np.cumsum(seg)))
    else:
        raise ValueError(f"Expected at least 3 columns, got {arr.shape[1]}")

    if len(x) < 2:
        raise ValueError("Need at least 2 points to create a trajectory")

    return x, y, z, md


def make_unique_well_name(base_name: str, existing_names: set[str]) -> str:
    """Ensure well name is unique by appending _N if needed."""
    if base_name not in existing_names:
        return base_name

    idx = 1
    while f"{base_name}_{idx}" in existing_names:
        idx += 1
    return f"{base_name}_{idx}"


# -------------------------------------------------
# Get trajectory files
# -------------------------------------------------
trajectory_files = discover_trajectory_files(mainpath)
print(f"Trajectory files found: {len(trajectory_files)}")

if not trajectory_files:
    print("No trajectory files found. Nothing to import.")


# -------------------------------------------------
# Import Wells into RMS
# -------------------------------------------------
existing_well_names = set(project.wells.names)

for filepath in trajectory_files:
    rel_name = os.path.relpath(filepath, mainpath)
    stem = os.path.splitext(rel_name)[0]
    base_well_name = stem.replace(os.sep, "_").replace(" ", "_")
    well_name = make_unique_well_name(base_well_name, existing_well_names)

    try:
        x, y, z, md = load_trajectory(filepath)
    except Exception as exc:
        print(f"Skipping {filepath}: {exc}")
        continue

    # Create well
    well = project.wells.create(well_name)
    existing_well_names.add(well_name)
    print(f"Well created: {well.name} (source: {filepath})")

    # Set wellhead and RKB
    well.wellhead = (float(x[0]), float(y[0]))
    well.rkb = float(-1.0 * z[0])

    # Create trajectory
    trajectories = well.wellbore.trajectories
    drilled_trajectory = trajectories.create("Drilled trajectory")

    surveypoints = drilled_trajectory.survey_point_series
    npts = len(md)
    array = surveypoints.generate_survey_points(npts)

    # RMS expected format
    array[:, 0] = md
    array[:, 3] = x
    array[:, 4] = y
    array[:, 5] = z

    surveypoints.set_survey_points(array)
    print(f"Trajectory imported for well {well_name}")

print("Import complete.")
