# Streamline performance and default parameters (2026-10-26)

## Fixed-parameter benchmark

One existing 18-frame aorta sample, canonical shape 74 x 112 x 110 x 18,
approximately 2.4 mm isotropic sampling. Grid seeds use 3 voxel widths, all
positive mask labels, seed-only mask confinement, both directions, midpoint
integration, 1 mm steps, the same 160-step cap and 200 mm length budget,
and a 0.1 source-unit stopping threshold. Both in-plane results and 3D slab
projections are computed for every spatial slice of every frame.

Sources were loaded once before timing. Times include content hashes,
integration, complete slice preindexing, and atomic disk-cache writes. The
same single-worker CPU configuration used OPENBLAS_NUM_THREADS=1 and
OMP_NUM_THREADS=1. GUI rendering and initial NRRD reading are excluded.

| Measurement | Baseline | Optimized |
| --- | ---: | ---: |
| All 18 frames + cache writes | 69.370 s | 39.817 s |
| Relative speed | 1.00 | 1.742 |
| Current-configuration memory | 196.637 MiB | 196.637 MiB |
| Disk cache size | 148.960 MiB | 152.006 MiB |

All **288 stored numeric arrays were bitwise identical**, including 3D traces,
velocities, slab segments/offsets, and every planar slice/offset. A separate
three-repeat peak-frame check reduced median computation from 4.464 s to
2.841 s. The initial cProfile run assigned about 84% of total time to planar
integration; those instrumented times were not used as speedup measurements.

A second server checked 24 synthetic combinations spanning oblique anisotropic
grids, midpoint/RK4, forward/backward/both directions, Grid/Random, and seed-only
versus trajectory mask confinement. Both 3D and all planar outputs were bitwise
identical to the frozen baseline.

The changes reuse already-interpolated accepted endpoints, pass explicit
interpolation buffers, omit unused trajectory-owner arrays, skip empty masked
planar slices and avoid unnecessary plane copies. Standard lossless NPZ
compression now uses level 1 rather than level 6. That makes cache writes
faster with about 2% larger files for this sample; old NPZ files remain readable.
No seeds, steps or stopping conditions were removed for this comparison.

## Default-parameter rationale and limits

These are interactive visualization starting values, not a clinically validated
or universally optimal configuration. Saved user overrides are retained.

| Parameter | New-profile default | Rationale |
| --- | --- | --- |
| Direction | Both | Shows upstream and downstream neighborhoods around each seed; arrows still follow velocity. |
| Integrator | Midpoint / RK2 | Lower sampling cost for interactive use; RK4 is available for curved-field checks. |
| Step | 1 mm | About 0.42 voxel for this 2.4 mm sample. It is an absolute physical setting, so it must be reconsidered for finer grids. |
| Maximum steps / direction | 200 | Matches current autoflow's configured cap. At 1 mm the length cap normally binds first. |
| Maximum complete length | 200 mm | Bounded display/computation budget, not an anatomical vessel-length estimate. Both directions receive 100 mm each. |
| Minimum length | 0 mm | Does not discard short vessels/trajectories by default. |
| Stop speed, calibrated | 1 cm/s | Matches autoflow's 0.01 m/s threshold after its cm/s-to-m/s conversion. Still needs adjustment to the actual noise and slow-flow regime. |
| Stop magnitude, unscaled | 0.1 source units | An engineering display threshold. It is not a physical noise estimate or a universal phase threshold. |
| Maximum turn | 180 degrees | No extra angular cutoff by default. |

For a full length budget L and step h, allowing at least ceil(L/(d*h)) steps per
direction avoids an earlier step-count stop; d is 2 for bidirectional tracing
and 1 otherwise. Reducing the step without increasing the cap can shorten the
traces even when Maximum length is unchanged.

On 12 fixed seeds from the same frame, RK2 with 1 mm steps was compared at common
arc lengths to a 0.25 mm RK4 numerical reference, preserving a 100 mm forward
length budget. The median per-seed 95th-percentile positional difference was
0.058 mm, but the worst seed reached 2.278 mm. With 0.5 mm RK2 steps these were
0.011 mm and 0.713 mm. This is a local step-convergence check, not ground-truth
velocity/anatomical accuracy, and does not establish performance across cases
or aliased phase fields. It supports retaining 1 mm as a preview setting while
making finer-step checks available where trajectories are sensitive.
