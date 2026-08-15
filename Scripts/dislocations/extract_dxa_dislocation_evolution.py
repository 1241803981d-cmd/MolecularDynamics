"""Extract HCP Ti dislocation evolution from LAMMPS compression trajectories.

The full Ti-Si-C frame is first Voronoi-partitioned. Atomic volumes belonging
to type-1 Ti atoms are summed to obtain the instantaneous Ti matrix volume.
Types 2 and 3 are then removed before HCP DXA is applied to the Ti matrix.
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import ovito
from ovito.io import import_file
from ovito.modifiers import (
    DeleteSelectedModifier,
    DislocationAnalysisModifier,
    ExpressionSelectionModifier,
    VoronoiAnalysisModifier,
)


SCRIPT_DIR = Path(__file__).resolve().parent

RATE_DIRS = {
    "0.25": "0p25",
    "0.5": "0p5",
    "1.0": "1p0",
    "2.0": "2p0",
    "5.0": "5p0",
}

TAG_PATTERN = re.compile(r"dump_compression_(R[^.]+)\.lammpstrj$")
DXA_PREFIX = "DislocationAnalysis.length."
DXA_LENGTH_KEYS = {
    "a_type_line_length_A": DXA_PREFIX + "1/3<11-20>",
    "ca_type_line_length_A": DXA_PREFIX + "1/3<11-23>",
    "c_type_line_length_A": DXA_PREFIX + "<0001>",
    "hcp_partial_line_length_A": DXA_PREFIX + "1/3<-1100>",
    "other_recognized_line_length_A": DXA_PREFIX + "<-1100>",
    "unclassified_line_length_A": DXA_PREFIX + "other",
}


def load_stress_table(path: Path) -> dict[int, dict[str, float]]:
    if not path.exists():
        return {}
    table = pd.read_csv(path, sep=r"\s+")
    return {
        int(row.Step): {
            "strain_lammps": float(row.Strain),
            "szz_raw_GPa": float(row.Szz_GPa),
            "compressive_stress_GPa": -float(row.Szz_GPa),
            "temperature_K": float(row.Temp),
        }
        for row in table.itertuples(index=False)
    }


def cell_geometry(data) -> tuple[float, float, float, float]:
    matrix = np.asarray(data.cell)[:, :3]
    lx, ly, lz = (float(np.linalg.norm(matrix[:, i])) for i in range(3))
    volume = float(abs(np.linalg.det(matrix)))
    return lx, ly, lz, volume


def attribute_float(data, key: str) -> float:
    return float(data.attributes.get(key, 0.0))


def density_m2(line_length_A: float, volume_A3: float) -> float:
    if volume_A3 <= 0:
        return np.nan
    return line_length_A / volume_A3 * 1.0e20


def build_pipeline(trajectory: Path):
    pipeline = import_file(str(trajectory), multiple_frames=True)
    pipeline.modifiers.append(VoronoiAnalysisModifier(compute_indices=False))
    pipeline.modifiers.append(
        ExpressionSelectionModifier(expression="ParticleType != 1")
    )
    pipeline.modifiers.append(DeleteSelectedModifier())
    dxa = DislocationAnalysisModifier(
        input_crystal_structure=DislocationAnalysisModifier.Lattice.HCP
    )
    dxa.trial_circuit_length = 14
    dxa.circuit_stretchability = 9
    dxa.only_perfect_dislocations = False
    # Keep raw line geometry for reproducible length measurements.
    dxa.line_smoothing_enabled = False
    dxa.line_coarsening_enabled = False
    pipeline.modifiers.append(dxa)
    return pipeline


def process_trajectory(
    trajectory: Path,
    stress_file: Path,
    cooling_rate: float,
    output_file: Path,
    frame_stride: int,
) -> dict[str, object]:
    match = TAG_PATTERN.search(trajectory.name)
    if not match:
        raise ValueError(f"Cannot parse trajectory tag: {trajectory.name}")
    tag = match.group(1)
    pipeline = build_pipeline(trajectory)
    stress_by_step = load_stress_table(stress_file)
    frame_indices = list(range(0, int(pipeline.source.num_frames), frame_stride))
    if frame_indices[-1] != int(pipeline.source.num_frames) - 1:
        frame_indices.append(int(pipeline.source.num_frames) - 1)

    rows: list[dict[str, object]] = []
    initial_lz: float | None = None
    initial_total_density: float | None = None
    start = time.perf_counter()

    for position, frame_index in enumerate(frame_indices, start=1):
        data = pipeline.compute(frame_index)
        step = int(data.attributes.get("Timestep", frame_index))
        lx, ly, lz, cell_volume = cell_geometry(data)
        if initial_lz is None:
            initial_lz = lz

        if "Atomic Volume" not in data.particles:
            raise RuntimeError("OVITO did not preserve the Voronoi Atomic Volume property")
        ti_volume = float(np.asarray(data.particles["Atomic Volume"]).sum())
        ti_count = int(data.particles.count)
        total_length = attribute_float(data, "DislocationAnalysis.total_line_length")
        total_density = density_m2(total_length, ti_volume)
        if initial_total_density is None:
            initial_total_density = total_density

        category_lengths = {
            column: attribute_float(data, key)
            for column, key in DXA_LENGTH_KEYS.items()
        }
        hcp_count = int(data.attributes.get("DislocationAnalysis.counts.HCP", 0))
        fcc_count = int(data.attributes.get("DislocationAnalysis.counts.FCC", 0))
        other_count = int(data.attributes.get("DislocationAnalysis.counts.OTHER", 0))
        stress_values = stress_by_step.get(step, {})
        strain_cell = (initial_lz - lz) / initial_lz
        row: dict[str, object] = {
            "cooling_rate_K_per_ps": cooling_rate,
            "tag": tag,
            "frame_index": frame_index,
            "step": step,
            "strain": stress_values.get("strain_lammps", strain_cell),
            "strain_from_cell": strain_cell,
            "szz_raw_GPa": stress_values.get("szz_raw_GPa", np.nan),
            "compressive_stress_GPa": stress_values.get(
                "compressive_stress_GPa", np.nan
            ),
            "temperature_K": stress_values.get("temperature_K", np.nan),
            "lx_A": lx,
            "ly_A": ly,
            "lz_A": lz,
            "cell_volume_A3": cell_volume,
            "ti_matrix_volume_A3": ti_volume,
            "ti_atom_count": ti_count,
            "total_line_length_A": total_length,
            "total_dislocation_density_m-2": total_density,
            "delta_total_dislocation_density_m-2": (
                total_density - initial_total_density
            ),
            "dislocation_segment_count": len(data.dislocations.lines),
            "hcp_atom_count": hcp_count,
            "fcc_atom_count": fcc_count,
            "other_structure_atom_count": other_count,
            "hcp_identified_fraction": hcp_count / ti_count if ti_count else np.nan,
            "other_structure_fraction": other_count / ti_count if ti_count else np.nan,
        }
        row.update(category_lengths)
        for length_column in DXA_LENGTH_KEYS:
            density_column = length_column.replace("line_length_A", "density_m-2")
            row[density_column] = density_m2(
                float(category_lengths[length_column]), ti_volume
            )
        if total_length > 0:
            unclassified_fraction = (
                float(category_lengths["unclassified_line_length_A"])
                / total_length
            )
            row["a_type_line_fraction"] = (
                float(category_lengths["a_type_line_length_A"]) / total_length
            )
            row["ca_type_line_fraction"] = (
                float(category_lengths["ca_type_line_length_A"]) / total_length
            )
        else:
            unclassified_fraction = 0.0
            row["a_type_line_fraction"] = 0.0
            row["ca_type_line_fraction"] = 0.0
        row["unclassified_line_fraction"] = unclassified_fraction
        if row["hcp_identified_fraction"] >= 0.5 and unclassified_fraction <= 0.5:
            row["dxa_quality_flag"] = "ACCEPTABLE"
        elif row["hcp_identified_fraction"] >= 0.3 and unclassified_fraction <= 0.7:
            row["dxa_quality_flag"] = "CAUTION"
        else:
            row["dxa_quality_flag"] = "LOW_CONFIDENCE"
        known_sum = float(sum(category_lengths.values()))
        row["line_length_balance_error_A"] = total_length - known_sum
        rows.append(row)

        elapsed = time.perf_counter() - start
        print(
            f"[{tag}] {position}/{len(frame_indices)} frame={frame_index} "
            f"strain={row['strain']:.4f} rho={total_density:.4e} m^-2 "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_file.with_suffix(".csv.tmp")
    pd.DataFrame(rows).to_csv(temporary, index=False)
    temporary.replace(output_file)
    return {
        "cooling_rate_K_per_ps": cooling_rate,
        "tag": tag,
        "trajectory": str(trajectory),
        "stress_file": str(stress_file),
        "frames_processed": len(rows),
        "output_csv": str(output_file),
        "status": "COMPLETED",
        "ovito_version": ovito.version_string,
    }


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
    parser.add_argument(
        "--rates",
        nargs="+",
        choices=list(RATE_DIRS),
        default=list(RATE_DIRS),
    )
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    if args.frame_stride < 1:
        parser.error("--frame-stride must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []
    missing_rates: list[str] = []

    for rate in args.rates:
        rate_dir = args.base_dir / RATE_DIRS[rate]
        trajectories = sorted(rate_dir.glob("dump_compression_R*.lammpstrj"))
        if not trajectories:
            missing_rates.append(rate)
            manifest.append(
                {
                    "cooling_rate_K_per_ps": float(rate),
                    "tag": "",
                    "trajectory": "",
                    "stress_file": "",
                    "frames_processed": 0,
                    "output_csv": "",
                    "status": "MISSING_TRAJECTORY",
                    "ovito_version": ovito.version_string,
                }
            )
            print(f"[rate {rate}] missing compression trajectory", flush=True)
            continue

        for trajectory in trajectories:
            match = TAG_PATTERN.search(trajectory.name)
            if not match:
                print(f"Skipping unrecognized name: {trajectory}")
                continue
            tag = match.group(1)
            stress_root = args.stress_dir if args.stress_dir is not None else rate_dir
            stress_file = stress_root / f"stress_strain_{tag}.dat"
            output_file = args.output_dir / f"dxa_dislocation_evolution_{tag}.csv"
            if output_file.exists() and not args.overwrite:
                manifest.append(
                    {
                        "cooling_rate_K_per_ps": float(rate),
                        "tag": tag,
                        "trajectory": str(trajectory),
                        "stress_file": str(stress_file),
                        "frames_processed": len(pd.read_csv(output_file)),
                        "output_csv": str(output_file),
                        "status": "SKIPPED_EXISTING",
                        "ovito_version": ovito.version_string,
                    }
                )
                print(f"[{tag}] output exists; use --overwrite to recompute")
                continue
            manifest.append(
                process_trajectory(
                    trajectory=trajectory,
                    stress_file=stress_file,
                    cooling_rate=float(rate),
                    output_file=output_file,
                    frame_stride=args.frame_stride,
                )
            )

    manifest_file = args.output_dir / "dxa_processing_manifest.csv"
    pd.DataFrame(manifest).to_csv(manifest_file, index=False)
    print(f"Wrote: {manifest_file.resolve()}")
    if missing_rates and args.strict:
        raise SystemExit(
            "Missing compression trajectories for rates: " + ", ".join(missing_rates)
        )


if __name__ == "__main__":
    main()
