"""Audit compression trajectories before OVITO DXA processing."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ovito.io import import_file


SCRIPT_DIR = Path(__file__).resolve().parent

RATE_DIRS = {
    "0.25": "0p25",
    "0.5": "0p5",
    "1.0": "1p0",
    "2.0": "2p0",
    "5.0": "5p0",
}


def cell_lengths(data) -> tuple[float, float, float]:
    matrix = np.asarray(data.cell)[:, :3]
    return tuple(float(np.linalg.norm(matrix[:, i])) for i in range(3))


def audit_rate(
    base_dir: Path,
    stress_dir: Path | None,
    rate: str,
    folder: str,
) -> dict[str, object]:
    rate_dir = base_dir / folder
    trajectories = sorted(rate_dir.glob("dump_compression_R*.lammpstrj"))
    stress_root = stress_dir if stress_dir is not None else rate_dir
    stress_files = sorted(stress_root.glob(f"stress_strain_R{folder}.dat"))
    row: dict[str, object] = {
        "cooling_rate_K_per_ps": float(rate),
        "rate_folder": folder,
        "trajectory_count": len(trajectories),
        "stress_file_count": len(stress_files),
        "trajectory": "",
        "stress_file": "",
        "frame_count": 0,
        "first_step": np.nan,
        "last_step": np.nan,
        "atom_count": np.nan,
        "ti_atom_count": np.nan,
        "si_atom_count": np.nan,
        "c_atom_count": np.nan,
        "initial_lz_A": np.nan,
        "final_lz_A": np.nan,
        "final_cell_strain": np.nan,
        "stress_final_strain": np.nan,
        "status": "MISSING_TRAJECTORY",
    }
    if not trajectories:
        return row

    trajectory = trajectories[0]
    row["trajectory"] = str(trajectory)
    pipeline = import_file(str(trajectory), multiple_frames=True)
    frame_count = int(pipeline.source.num_frames)
    first = pipeline.compute(0)
    last = pipeline.compute(frame_count - 1)
    first_lz = cell_lengths(first)[2]
    last_lz = cell_lengths(last)[2]
    particle_types = np.asarray(first.particles["Particle Type"])
    row.update(
        {
            "frame_count": frame_count,
            "first_step": int(first.attributes.get("Timestep", 0)),
            "last_step": int(last.attributes.get("Timestep", 0)),
            "atom_count": int(first.particles.count),
            "ti_atom_count": int(np.count_nonzero(particle_types == 1)),
            "si_atom_count": int(np.count_nonzero(particle_types == 2)),
            "c_atom_count": int(np.count_nonzero(particle_types == 3)),
            "initial_lz_A": first_lz,
            "final_lz_A": last_lz,
            "final_cell_strain": (first_lz - last_lz) / first_lz,
        }
    )

    if stress_files:
        stress_file = stress_files[0]
        row["stress_file"] = str(stress_file)
        stress = pd.read_csv(stress_file, sep=r"\s+")
        row["stress_final_strain"] = float(stress["Strain"].iloc[-1])
        complete = (
            frame_count >= 41
            and abs(float(row["final_cell_strain"]) - 0.4) <= 0.01
            and abs(float(row["stress_final_strain"]) - 0.4) <= 0.001
        )
        row["status"] = "READY" if complete else "INCOMPLETE"
    else:
        row["status"] = "MISSING_STRESS_FILE"
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=SCRIPT_DIR,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results",
    )
    parser.add_argument("--stress-dir", type=Path, default=None)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        audit_rate(args.base_dir, args.stress_dir, rate, folder)
        for rate, folder in RATE_DIRS.items()
    ]
    output = args.output_dir / "trajectory_input_audit.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print(f"\nWrote: {output.resolve()}")


if __name__ == "__main__":
    main()
