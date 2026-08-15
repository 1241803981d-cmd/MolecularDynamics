# Ti/SiC cooling
A repository for molecular dynamics simulations.
# Ti/SiC Cooling-Rate-Dependent Compression Data and Code

This repository contains simulation inputs, numerical source data, final atomic
configurations, and post-processing scripts associated with the manuscript:

> Atomic-scale mechanisms of cooling-rate-dependent compression behavior in SiC particle reinforced titanium matrix composites


LAMMPS atom types are defined as follows:

| Atom type | Element |
| 1 | Ti |
| 2 | Si |
| 3 | C |

The velocity initialization value used by the input script is 4652817.

## Repository contents

```text
.
|-- README.md
|-- LICENSE
|-- requirements.txt
|-- simulation/
|   |-- in.lmp
|   |-- Ti-SiC.data
|   `-- potentials/
|       |-- Ti.eam.alloy
|       `-- SiC.vashishta
|-- data/
|   |-- cooling/
|   |-- stress_strain/
|   |-- rdf/
|   |-- vdos/
|   |-- dislocations/
|   `-- final_atomic_states/
`-- scripts/
    |-- vdos/
    |-- dislocations/
    `-- stress_maps/
```

## Simulation inputs

- `simulation/in.lmp`: minimization, NPT equilibration, controlled cooling,
  interface VACF sampling, and uniaxial compression.
- `simulation/Ti-SiC.data`: initial 59,054-atom Ti/SiC model.
- `simulation/potentials/`: Ti EAM and SiC Vashishta potential files.
- Ti-Si and Ti-C cross interactions are defined by the Morse parameters in
  `simulation/in.lmp`.

## Numerical source data

- `data/cooling/`: temperature, potential energy, kinetic energy, pressure,
  cell dimensions, and volume during equilibration and cooling.
- `data/stress_strain/`: compressive strain, stress tensor components,
  equivalent stress, temperature, energy, and cell dimensions.
- `data/rdf/`: Ti-Si, Ti-C, and Si-C radial distribution functions.
- `data/vdos/`: interface-Ti velocity autocorrelation functions and normalized
  vibrational density of states.
- `data/dislocations/`: total, `<a>`-type, and `<c+a>`-type dislocation evolution
  obtained with OVITO DXA.
- `data/final_atomic_states/`: final compressed configurations and local
  von Mises stress slice statistics.

Cooling-rate tags use `p` as the decimal separator. For example, `R0p25`
denotes 0.25 K/ps and `R5p0` denotes 5.0 K/ps.

## Software environment

The local environment used to prepare and verify the repository is:

| Software | Version |
| LAMMPS | 22 Jul 2025, Update 1 |
| Python | 3.13.5 |
| OVITO Python module | 3.14.1 |
| NumPy | 2.3.4 |
| Pandas | 2.3.3 |
| SciPy | 1.16.3 |
| Matplotlib | 3.10.7 |

The LAMMPS executable is compiled for Windows with Microsoft MPI
10.1.12498.18 and OpenMP 4.5.

## Running the simulations

Run LAMMPS from the `simulation` directory. Select the cooling rate and output
tag on the command line:

```powershell
cd simulation
lmp -in in.lmp -var coolRate 0.25 -var tag R0p25
lmp -in in.lmp -var coolRate 0.5  -var tag R0p5
lmp -in in.lmp -var coolRate 1.0  -var tag R1p0
lmp -in in.lmp -var coolRate 2.0  -var tag R2p0
lmp -in in.lmp -var coolRate 5.0  -var tag R5p0
```

The input script uses metal units, a timestep of 0.001 ps, a cooling interval
of 400 K, and automatically calculates the number of cooling steps from the
selected cooling rate.

# Post-processing

## VDOS from interface VACF

```powershell
cd data/vdos
python ../../scripts/vdos/compute_vdos_from_vacf.py
```

The script applies VACF normalization, a Hann window, fivefold zero padding,
and a real-valued fast Fourier transform. It exports
`VDOS_Data_for_Origin.csv` over 0-30 THz.

### Final local von Mises stress slices

```powershell
python scripts/stress_maps/plot_final_compressed_vm_stress_slices.py `
  --dump-dir data/final_atomic_states `
  --out-dir figures/stress_maps
```

The default slice thickness is 3.0 angstrom and the Ti atomic volume used for
stress normalization is 17.6 angstrom^3 per atom.

### Dislocation analysis

The DXA workflow and commands are documented in
`scripts/dislocations/README.md`. The analysis uses an HCP reference lattice,
Voronoi atomic volumes, and OVITO 3.14.1.

## Interatomic-potential references

The Ti EAM potential cites X. W. Zhou, R. A. Johnson, and H. N. G. Wadley,
Physical Review B 69, 144113 (2004).

The SiC potential cites P. Vashishta, R. K. Kalia, A. Nakano, and J. P. Rino,
Journal of Applied Physics 101, 103515 (2007).

## License

Code and documentation are distributed under the MIT License. Numerical data
are distributed under the Creative Commons Attribution 4.0 International
license. See `LICENSE` for details.
