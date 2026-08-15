"""Combine per-trajectory DXA CSV files into manuscript-ready tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent

DENSITY_COLUMNS = [
    "cooling_rate_K_per_ps",
    "tag",
    "step",
    "strain",
    "compressive_stress_GPa",
    "total_line_length_A",
    "ti_matrix_volume_A3",
    "total_dislocation_density_m-2",
    "delta_total_dislocation_density_m-2",
    "hcp_identified_fraction",
    "other_structure_fraction",
    "unclassified_line_fraction",
    "dxa_quality_flag",
]

TYPE_COLUMNS = [
    "cooling_rate_K_per_ps",
    "tag",
    "step",
    "strain",
    "compressive_stress_GPa",
    "a_type_line_length_A",
    "a_type_density_m-2",
    "ca_type_line_length_A",
    "ca_type_density_m-2",
    "c_type_line_length_A",
    "c_type_density_m-2",
    "hcp_partial_line_length_A",
    "hcp_partial_density_m-2",
    "unclassified_line_length_A",
    "unclassified_density_m-2",
    "a_type_line_fraction",
    "ca_type_line_fraction",
    "unclassified_line_fraction",
    "dxa_quality_flag",
]


def summarize(group: pd.DataFrame) -> dict[str, object]:
    group = group.sort_values("strain")
    peak_index = group["compressive_stress_GPa"].idxmax()
    peak = group.loc[peak_index]
    final = group.iloc[-1]
    max_density = group.loc[group["total_dislocation_density_m-2"].idxmax()]
    return {
        "cooling_rate_K_per_ps": float(group["cooling_rate_K_per_ps"].iloc[0]),
        "tag": group["tag"].iloc[0],
        "initial_total_density_m-2": float(
            group["total_dislocation_density_m-2"].iloc[0]
        ),
        "peak_stress_GPa": float(peak["compressive_stress_GPa"]),
        "strain_at_peak_stress": float(peak["strain"]),
        "total_density_at_peak_stress_m-2": float(
            peak["total_dislocation_density_m-2"]
        ),
        "a_type_density_at_peak_stress_m-2": float(peak["a_type_density_m-2"]),
        "ca_type_density_at_peak_stress_m-2": float(peak["ca_type_density_m-2"]),
        "maximum_total_density_m-2": float(
            max_density["total_dislocation_density_m-2"]
        ),
        "strain_at_maximum_density": float(max_density["strain"]),
        "final_strain": float(final["strain"]),
        "final_total_density_m-2": float(final["total_dislocation_density_m-2"]),
        "final_a_type_density_m-2": float(final["a_type_density_m-2"]),
        "final_ca_type_density_m-2": float(final["ca_type_density_m-2"]),
        "minimum_hcp_identified_fraction": float(
            group["hcp_identified_fraction"].min()
        ),
        "maximum_unclassified_line_fraction": float(
            np.divide(
                group["unclassified_line_length_A"],
                group["total_line_length_A"],
                out=np.zeros(len(group), dtype=float),
                where=group["total_line_length_A"].to_numpy() > 0,
            ).max()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=SCRIPT_DIR / "results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results",
    )
    args = parser.parse_args()

    files = sorted(args.input_dir.glob("dxa_dislocation_evolution_R*.csv"))
    if not files:
        raise SystemExit(f"No per-trajectory DXA CSV files found in {args.input_dir}")
    combined = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
    combined = combined.sort_values(
        ["cooling_rate_K_per_ps", "strain"]
    ).reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    density_file = args.output_dir / "total_dislocation_density_all_rates.csv"
    types_file = args.output_dir / "a_and_ca_dislocation_evolution_all_rates.csv"
    summary_file = args.output_dir / "dislocation_summary_by_rate.csv"
    combined[DENSITY_COLUMNS].to_csv(density_file, index=False)
    combined[TYPE_COLUMNS].to_csv(types_file, index=False)
    summary = pd.DataFrame(
        summarize(group)
        for _, group in combined.groupby("cooling_rate_K_per_ps", sort=True)
    )
    summary.to_csv(summary_file, index=False)
    print(f"Wrote: {density_file.resolve()}")
    print(f"Wrote: {types_file.resolve()}")
    print(f"Wrote: {summary_file.resolve()}")


if __name__ == "__main__":
    main()
