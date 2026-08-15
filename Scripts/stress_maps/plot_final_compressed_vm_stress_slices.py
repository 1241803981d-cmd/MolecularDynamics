"""
Plot local von Mises equivalent stress slices from final compressed Ti/SiC dumps.

This script is intended for files named like:
    atomic_features_final_R0p5.dump

Compared with a simple 0-100 A slice script, this version:
1. Reads the actual deformed box bounds from each dump file.
2. Computes the slice center from the actual SiC atoms in that dump.
3. Uses the final compressed configuration as the plotted state.
4. Exports both individual figures and a shared-colorbar comparison figure.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_DUMP_PATTERN = "atomic_features_final_*.dump"
DEFAULT_ATOMIC_VOLUME_TI = 17.6  # A^3 per Ti atom, used to convert stress*volume to stress
BAR_TO_GPA = 1.0e-4


def configure_matplotlib() -> None:
    mpl.rcParams["font.family"] = "serif"
    mpl.rcParams["font.serif"] = ["Times New Roman"]
    mpl.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams["font.size"] = 16
    mpl.rcParams["axes.labelsize"] = 22
    mpl.rcParams["xtick.labelsize"] = 17
    mpl.rcParams["ytick.labelsize"] = 17
    mpl.rcParams["legend.fontsize"] = 16
    mpl.rcParams["axes.unicode_minus"] = False


def parse_dump(path: Path) -> tuple[int, list[tuple[float, float]], pd.DataFrame]:
    """Read a single-frame LAMMPS custom dump."""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        header = [next(f).strip() for _ in range(9)]

    if header[0] != "ITEM: TIMESTEP":
        raise ValueError(f"{path.name}: not a LAMMPS custom dump")

    timestep = int(header[1])
    natoms = int(header[3])
    bounds = []
    for line in header[5:8]:
        lo, hi = [float(x) for x in line.split()[:2]]
        bounds.append((lo, hi))

    atom_columns = header[8].replace("ITEM: ATOMS", "").split()
    df = pd.read_csv(path, sep=r"\s+", skiprows=9, names=atom_columns, engine="python")
    if len(df) != natoms:
        raise ValueError(f"{path.name}: expected {natoms} atoms, read {len(df)}")

    return timestep, bounds, df


def rate_label_from_name(name: str) -> str:
    match = re.search(r"_R([0-9]+p?[0-9]*)_", name)
    if not match:
        return name.replace(".dump", "")
    value = match.group(1).replace("p", ".")
    return f"{float(value):g} K/ps"


def tag_from_name(name: str) -> str:
    match = re.search(r"atomic_features_final_(.+)\.dump$", name)
    return match.group(1) if match else name.replace(".dump", "")


def add_vm_stress_gpa(df: pd.DataFrame, atomic_volume_ti: float) -> pd.DataFrame:
    df = df.copy()
    if "v_vm_raw" in df.columns:
        df["vm_GPa"] = (df["v_vm_raw"] / atomic_volume_ti) * BAR_TO_GPA
        return df

    required = [
        "c_s_atom[1]",
        "c_s_atom[2]",
        "c_s_atom[3]",
        "c_s_atom[4]",
        "c_s_atom[5]",
        "c_s_atom[6]",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Cannot compute von Mises stress; missing columns: {missing}")

    sx = df["c_s_atom[1]"]
    sy = df["c_s_atom[2]"]
    sz = df["c_s_atom[3]"]
    sxy = df["c_s_atom[4]"]
    sxz = df["c_s_atom[5]"]
    syz = df["c_s_atom[6]"]
    vm_raw = np.sqrt(
        0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
        + 3.0 * (sxy**2 + sxz**2 + syz**2)
    )
    df["vm_GPa"] = (vm_raw / atomic_volume_ti) * BAR_TO_GPA
    return df


def get_sic_center(df: pd.DataFrame) -> np.ndarray:
    sic = df[df["type"].isin([2, 3])]
    if sic.empty:
        raise ValueError("No SiC atoms found; expected atom types 2 and 3")
    return sic[["x", "y", "z"]].mean().to_numpy()


def slice_atoms(df: pd.DataFrame, z_center: float, thickness: float) -> pd.DataFrame:
    half = thickness / 2.0
    return df[(df["z"] >= z_center - half) & (df["z"] <= z_center + half)].copy()


def plot_one(
    ax,
    df_slice: pd.DataFrame,
    bounds: list[tuple[float, float]],
    label: str,
    vmin: float,
    vmax: float,
    point_size: float,
    sic_point_size: float,
    cmap: str,
    show_xlabel: bool = True,
    show_ylabel: bool = True,
):
    ti = df_slice[df_slice["type"] == 1]
    sic = df_slice[df_slice["type"].isin([2, 3])]

    sc = ax.scatter(
        ti["x"],
        ti["y"],
        c=ti["vm_GPa"],
        cmap=cmap,
        s=point_size,
        alpha=0.92,
        vmin=vmin,
        vmax=vmax,
        linewidths=0,
        rasterized=True,
        zorder=2,
    )
    ax.scatter(
        sic["x"],
        sic["y"],
        color="#404040",
        s=sic_point_size,
        alpha=1.0,
        linewidths=0,
        label="SiC particle",
        rasterized=True,
        zorder=3,
    )

    (xlo, xhi), (ylo, yhi), _ = bounds
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"X coordinate ($\AA$)" if show_xlabel else "")
    ax.set_ylabel(r"Y coordinate ($\AA$)" if show_ylabel else "")
    ax.tick_params(direction="in")
    return sc


def process_dump(
    dump_path: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict:
    timestep, bounds, df = parse_dump(dump_path)
    df = add_vm_stress_gpa(df, args.atomic_volume_ti)

    sic_center = get_sic_center(df)
    df_slice = slice_atoms(df, z_center=sic_center[2], thickness=args.slice_thickness)

    if df_slice.empty:
        raise ValueError(f"{dump_path.name}: slice contains no atoms")

    label = rate_label_from_name(dump_path.name)
    tag = tag_from_name(dump_path.name)

    fig, ax = plt.subplots(figsize=(8.0, 7.2))
    sc = plot_one(
        ax,
        df_slice,
        bounds,
        label=f"{label}, final compression",
        vmin=args.vmin,
        vmax=args.vmax,
        point_size=args.point_size,
        sic_point_size=args.sic_point_size,
        cmap=args.cmap,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Von Mises equivalent stress (GPa)", rotation=270, labelpad=22)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    png_path = out_dir / f"Stress_Contour_Final_{tag}.png"
    fig.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    ti = df_slice[df_slice["type"] == 1]
    sic = df_slice[df_slice["type"].isin([2, 3])]
    return {
        "file": dump_path.name,
        "tag": tag,
        "label": label,
        "timestep": timestep,
        "xlo": bounds[0][0],
        "xhi": bounds[0][1],
        "ylo": bounds[1][0],
        "yhi": bounds[1][1],
        "zlo": bounds[2][0],
        "zhi": bounds[2][1],
        "sic_center_x": sic_center[0],
        "sic_center_y": sic_center[1],
        "sic_center_z": sic_center[2],
        "slice_thickness_A": args.slice_thickness,
        "slice_atoms": len(df_slice),
        "slice_ti_atoms": len(ti),
        "slice_sic_atoms": len(sic),
        "ti_vm_mean_GPa": ti["vm_GPa"].mean(),
        "ti_vm_p95_GPa": ti["vm_GPa"].quantile(0.95),
        "ti_vm_max_GPa": ti["vm_GPa"].max(),
        "png": str(png_path),
    }


def plot_comparison(
    dump_paths: list[Path],
    out_dir: Path,
    args: argparse.Namespace,
) -> Path:
    n = len(dump_paths)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 5.2 * nrows))
    axes = np.asarray(axes).reshape(-1)
    last_sc = None

    for i, (ax, dump_path) in enumerate(zip(axes, dump_paths)):
        _, bounds, df = parse_dump(dump_path)
        df = add_vm_stress_gpa(df, args.atomic_volume_ti)
        sic_center = get_sic_center(df)
        df_slice = slice_atoms(df, z_center=sic_center[2], thickness=args.slice_thickness)
        row = i // ncols
        col = i % ncols
        last_sc = plot_one(
            ax,
            df_slice,
            bounds,
            label=rate_label_from_name(dump_path.name),
            vmin=args.vmin,
            vmax=args.vmax,
            point_size=args.point_size * 0.78,
            sic_point_size=args.sic_point_size * 0.78,
            cmap=args.cmap,
            show_xlabel=(row == nrows - 1),
            show_ylabel=(col == 0),
        )
        ax.legend().remove()

    for ax in axes[n:]:
        ax.axis("off")

    cbar = fig.colorbar(last_sc, ax=axes[:n], fraction=0.025, pad=0.02)
    cbar.set_label("Von Mises equivalent stress (GPa)", rotation=270, labelpad=22)
    fig.suptitle(
        f"Final compressed-state local stress slices, thickness = {args.slice_thickness:g} $\\AA$",
        y=0.985,
        fontsize=17,
    )
    fig.subplots_adjust(left=0.06, right=0.88, bottom=0.07, top=0.91, wspace=0.28, hspace=0.34)
    fig.savefig(out_dir / "Stress_Contour_Final_AllRates.png", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return out_dir / "Stress_Contour_Final_AllRates.png"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot final compressed-state local von Mises stress slices."
    )
    parser.add_argument("--dump-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--pattern", default=DEFAULT_DUMP_PATTERN)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--slice-thickness", type=float, default=3.0)
    parser.add_argument("--atomic-volume-ti", type=float, default=DEFAULT_ATOMIC_VOLUME_TI)
    parser.add_argument("--vmin", type=float, default=0.0)
    parser.add_argument("--vmax", type=float, default=35.0)
    parser.add_argument("--point-size", type=float, default=15.0)
    parser.add_argument("--sic-point-size", type=float, default=22.0)
    parser.add_argument("--cmap", default="jet")
    parser.add_argument("--dpi", type=int, default=600)
    args = parser.parse_args()

    configure_matplotlib()

    dump_dir = args.dump_dir.resolve()
    out_dir = args.out_dir or dump_dir / "final_vm_stress_slices"
    out_dir.mkdir(parents=True, exist_ok=True)

    dump_paths = sorted(dump_dir.glob(args.pattern))
    if not dump_paths:
        raise FileNotFoundError(f"No dump files found in {dump_dir} with pattern {args.pattern}")

    summary = []
    for dump_path in dump_paths:
        print(f"Processing {dump_path.name} ...")
        summary.append(process_dump(dump_path, out_dir, args))

    comparison = plot_comparison(dump_paths, out_dir, args)

    summary_path = out_dir / "final_vm_stress_slice_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)

    print("\nDone.")
    print(f"Individual figures: {out_dir}")
    print(f"Comparison figure: {comparison}")
    print(f"Summary CSV: {summary_path}")


if __name__ == "__main__":
    main()
