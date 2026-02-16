import os
import numpy as np

"""
Sidetrack wellpath planner.

Behavior:
1) Follow an existing trajectory exactly up to KOP.
2) From KOP onward, steer through WELL_GUIDE cells.
3) Enforce a maximum dogleg severity (deg/30m).
4) Bias the vertical position inside each guide window with structure_bias in [0,1]:
      0.0 = absolute top of the local WELL_GUIDE window
      1.0 = absolute bottom of the local WELL_GUIDE window

This script is designed for RMS Python environments where `project` is available.
"""


# -----------------------------
# USER INPUTS
# -----------------------------
grid_model_name = "Restart"
property_name = "WELL_GUIDE"
realisation = 0
output_filename = "../../rms/input/well_modelling/Target_sidetrack.txt"

existing_path_file = "../../rms/input/well_modelling/existing_path.txt"  # XYZ columns
kop_md = 1800.0                # MD (m) on existing path where sidetrack starts
step_length = 30.0             # trajectory step (m)
max_dls = 3.0                  # deg per 30 m
num_bins = 50                  # guide corridor stations
min_cells_per_bin = 3
structure_bias = 0.15          # 0=top of guide, 1=bottom of guide


# -----------------------------
# Geometry helpers
# -----------------------------
def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def rotate_towards(current_dir: np.ndarray, desired_dir: np.ndarray, max_turn_deg: float) -> np.ndarray:
    """Rotate current_dir towards desired_dir by at most max_turn_deg."""
    a = normalize(current_dir)
    b = normalize(desired_dir)

    c = np.clip(np.dot(a, b), -1.0, 1.0)
    angle = np.arccos(c)
    max_turn = np.radians(max_turn_deg)

    if angle < 1e-12:
        return b
    if angle <= max_turn:
        return b

    axis = np.cross(a, b)
    axis_n = np.linalg.norm(axis)
    if axis_n < 1e-12:
        ortho = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ortho, a)) > 0.9:
            ortho = np.array([0.0, 1.0, 0.0])
        axis = normalize(np.cross(a, ortho))
    else:
        axis = axis / axis_n

    theta = max_turn
    return (
        a * np.cos(theta)
        + np.cross(axis, a) * np.sin(theta)
        + axis * np.dot(axis, a) * (1.0 - np.cos(theta))
    )


def cumulative_md(points_xyz: np.ndarray) -> np.ndarray:
    d = np.linalg.norm(np.diff(points_xyz, axis=0), axis=1)
    return np.concatenate(([0.0], np.cumsum(d)))


def interpolate_point_at_md(points_xyz: np.ndarray, md: np.ndarray, md_target: float) -> np.ndarray:
    out = np.zeros(3, dtype=float)
    for i in range(3):
        out[i] = np.interp(md_target, md, points_xyz[:, i])
    return out


def direction_at_md(points_xyz: np.ndarray, md: np.ndarray, md_target: float) -> np.ndarray:
    if md_target <= md[0]:
        return normalize(points_xyz[1] - points_xyz[0])
    if md_target >= md[-1]:
        return normalize(points_xyz[-1] - points_xyz[-2])

    idx = np.searchsorted(md, md_target)
    i0 = max(0, idx - 1)
    i1 = min(len(points_xyz) - 1, idx)
    if i1 == i0:
        i1 = min(len(points_xyz) - 1, i0 + 1)
    return normalize(points_xyz[i1] - points_xyz[i0])


# -----------------------------
# Guide extraction
# -----------------------------
def extract_wellguide_xyz(project_obj) -> np.ndarray:
    grid_model = project_obj.grid_models[grid_model_name]
    grid = grid_model.get_grid(realisation)
    values = grid_model.properties[property_name].get_values(realisation)

    selected_xyz = []
    for cell_no, value in enumerate(values):
        if value == 1:
            selected_xyz.append(grid.get_cell_centers(cell_no))

    selected_xyz = np.array(selected_xyz, dtype=float)
    if selected_xyz.shape[0] == 0:
        raise RuntimeError("No WELL_GUIDE cells with value = 1.")
    return selected_xyz


def build_guide_control_points(selected_xyz: np.ndarray, bins_count: int, bias: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if selected_xyz.shape[0] < 5:
        raise RuntimeError("Not enough WELL_GUIDE cells to define a corridor.")

    centroid = selected_xyz.mean(axis=0)
    centered = selected_xyz - centroid

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_axis = eigvecs[:, np.argmax(eigvals)]
    major_axis[2] = 0.0
    major_axis = normalize(major_axis)

    projections = centered.dot(major_axis)
    order = np.argsort(projections)
    sorted_xyz = selected_xyz[order]
    sorted_proj = projections[order]

    bins = np.linspace(sorted_proj.min(), sorted_proj.max(), bins_count)

    cps = []
    cps_proj = []
    for i in range(len(bins) - 1):
        mask = (sorted_proj >= bins[i]) & (sorted_proj < bins[i + 1])
        if np.sum(mask) < min_cells_per_bin:
            continue

        chunk = sorted_xyz[mask]
        x = np.mean(chunk[:, 0])
        y = np.mean(chunk[:, 1])

        zmin = np.min(chunk[:, 2])
        zmax = np.max(chunk[:, 2])
        z = zmin + np.clip(bias, 0.0, 1.0) * (zmax - zmin)

        cps.append([x, y, z])
        cps_proj.append(np.mean(sorted_proj[mask]))

    control_points = np.array(cps, dtype=float)
    cp_proj = np.array(cps_proj, dtype=float)

    if len(control_points) < 2:
        raise RuntimeError("Could not build enough guide control points.")

    return control_points, major_axis, cp_proj


# -----------------------------
# Sidetrack planner
# -----------------------------
def plan_sidetrack(existing_xyz: np.ndarray, control_points: np.ndarray, major_axis: np.ndarray, cp_proj: np.ndarray) -> np.ndarray:
    md_existing = cumulative_md(existing_xyz)
    if kop_md < md_existing[0] or kop_md > md_existing[-1]:
        raise ValueError(f"kop_md={kop_md:.1f} is outside existing MD range [0, {md_existing[-1]:.1f}].")

    kop_xyz = interpolate_point_at_md(existing_xyz, md_existing, kop_md)
    init_dir = direction_at_md(existing_xyz, md_existing, kop_md)

    prefix_mask = md_existing < kop_md
    prefix = existing_xyz[prefix_mask]
    if len(prefix) == 0:
        prefix = np.array([existing_xyz[0]])
    prefix = np.vstack([prefix, kop_xyz])

    projected = (control_points - kop_xyz).dot(major_axis)
    forward_mask = projected > -step_length
    forward_cp = control_points[forward_mask]
    forward_proj = cp_proj[forward_mask]

    if len(forward_cp) < 2:
        raise RuntimeError("Not enough forward guide points after KOP.")

    total_proj = forward_proj.max() - forward_proj.min()
    n_steps = int(max(20, np.ceil((total_proj + 600.0) / step_length)))

    max_turn_deg = max_dls * (step_length / 30.0)

    sidetrack = [kop_xyz]
    current = kop_xyz.copy()
    current_dir = init_dir.copy()

    for _ in range(n_steps):
        p = (current - kop_xyz).dot(major_axis)
        target_idx = int(np.argmin(np.abs(forward_proj - p)))
        look_ahead_idx = min(target_idx + 2, len(forward_cp) - 1)
        target = forward_cp[look_ahead_idx]

        desired = target - current
        if np.linalg.norm(desired) < 1e-6:
            break

        new_dir = rotate_towards(current_dir, desired, max_turn_deg)
        current = current + step_length * new_dir

        sidetrack.append(current.copy())
        current_dir = new_dir

        if look_ahead_idx >= len(forward_cp) - 1 and np.linalg.norm(current - forward_cp[-1]) < 2.0 * step_length:
            break

    sidetrack = np.array(sidetrack)
    return np.vstack([prefix[:-1], sidetrack])


def load_existing_path(path: str) -> np.ndarray:
    xyz = np.loadtxt(path, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] < 3:
        raise ValueError("existing_path_file must contain at least 3 columns (X Y Z).")
    return xyz[:, :3]


# -----------------------------
# Main
# -----------------------------
existing_xyz = load_existing_path(existing_path_file)
selected_xyz = extract_wellguide_xyz(project)
control_points, major_axis, cp_proj = build_guide_control_points(selected_xyz, num_bins, structure_bias)
trajectory = plan_sidetrack(existing_xyz, control_points, major_axis, cp_proj)

output_path = os.path.join(os.getcwd(), output_filename)
os.makedirs(os.path.dirname(output_path), exist_ok=True)

with open(output_path, "w") as f:
    for p in trajectory:
        f.write(f"{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}\n")

print("Sidetrack exported:", output_path)
print(f"Structure bias used: {structure_bias:.2f} (0=top, 1=bottom of guide window)")
