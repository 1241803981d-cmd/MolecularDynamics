# Ti-SiC compression dislocation analysis

These scripts use OVITO's Dislocation Extraction Algorithm (DXA) with an HCP
reference lattice. LAMMPS atom type 1 is treated as Ti; types 2 and 3 are removed
before DXA. The instantaneous Ti matrix volume is the sum of type-1 Voronoi
atomic volumes computed while all Ti, Si, and C atoms are still present.

## Run order

```powershell
cd scripts/dislocations

python audit_dislocation_inputs.py --base-dir ../../trajectories --stress-dir ../../data/stress_strain --output-dir ../../data/dislocations

python extract_dxa_dislocation_evolution.py --base-dir ../../trajectories --stress-dir ../../data/stress_strain --output-dir ../../data/dislocations --strict

python combine_dxa_dislocation_csvs.py --input-dir ../../data/dislocations --output-dir ../../data/dislocations
```

The scripts use their own directory as the input root. The five cooling-rate
subdirectories must therefore be located beside the scripts. All CSV outputs are
written to the `results` subdirectory.

Use `--frame-stride 2` for a quicker exploratory run. Use the default stride of
1 for manuscript data. Existing per-trajectory outputs are not overwritten unless
`--overwrite` is supplied.

## Burgers-vector categories

- `a_type`: OVITO `1/3<11-20>`, the standard HCP `<a>` family.
- `ca_type`: OVITO `1/3<11-23>`, the HCP `<c+a>` family.
- `c_type`: OVITO `<0001>`.
- `hcp_partial`: OVITO `1/3<-1100>`; retained separately and not merged into
  the standard `<a>` family.
- `unclassified`: DXA lines reported as `other`.

The CSV files include the HCP-identified atom fraction and unclassified line
length. These diagnostics should be reported or checked because DXA reliability
can decrease near the SiC/Ti interface and at large compressive strains.

`dxa_quality_flag` is an explicit screening aid:

- `ACCEPTABLE`: HCP fraction is at least 0.5 and unclassified line fraction is
  at most 0.5.
- `CAUTION`: HCP fraction is at least 0.3 and unclassified line fraction is at
  most 0.7.
- `LOW_CONFIDENCE`: neither condition is met.

These are transparent project-level thresholds, not universal OVITO criteria.
Sensitivity checks and atomic snapshots remain necessary for interpretation.

## Main outputs

- `trajectory_input_audit.csv`: input availability and trajectory completeness.
- `dxa_dislocation_evolution_<tag>.csv`: complete frame-by-frame output for one
  cooling rate.
- `dxa_processing_manifest.csv`: processed, skipped, and missing trajectories.
- `total_dislocation_density_all_rates.csv`: total line length divided by the
  instantaneous Ti matrix volume.
- `a_and_ca_dislocation_evolution_all_rates.csv`: `<a>` and `<c+a>` line lengths
  and densities versus strain.
- `dislocation_summary_by_rate.csv`: peak-stress, maximum-density, and final
  state descriptors for manuscript discussion.

Line length is in angstrom, volume in angstrom cubed, and dislocation density in
inverse square metres. `delta_total_dislocation_density_m-2` subtracts the first
compression frame and is useful for separating compression-generated defects from
interface-related DXA lines already present before loading.
