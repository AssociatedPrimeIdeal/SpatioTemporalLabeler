# Streamline overlays

Open **View > Streamlines** or the toolbar **Streamlines** button. Enable overlays
and select three loaded scalar image sequences. The panel is organized into
**Inputs**, **Seeds**, **Trace**, **2D**, and **Display** tabs, with computation
status and cache actions always visible below. Only parameters active for the current choices are shown; values are retained while hidden. Each parameter has an English tooltip. The interface is English only; legacy language settings migrate automatically.

## Inputs

| Parameter | Meaning |
| --- | --- |
| Components 1/2/3 | Scalar sequences used for the vector components; defaults are flow_x, flow_y, flow_z. Their voxel grids must match; a 3D component can broadcast over time. |
| Positive direction | Anatomical direction of a positive component, independent of display flips. Segdata defaults are F/P/L (HF/AP/RL), rather than RAS X/Y/Z. |
| Input unit | Source values are unscaled; cm/s and m/s represent calibrated velocity. Phase (rad) + VENC converts each component using phase * VENC / pi. |
| VENC x/y/z | The VENC in cm/s for each selected component, required only for phase input. All three must be positive. |

The three anatomical components must cover R/L, A/P and H/F once each. Existing
preparation outputs mix phase radians and calibrated velocity, so filenames do
not establish units. Thresholds and colors use source magnitude in unscaled
mode, and cm/s in calibrated modes. The module does not unwrap aliased phase.

## Seeds

| Parameter | Meaning |
| --- | --- |
| Seed mask | Choose a loaded label sequence to seed only inside it, or use the full field. A matching 3D mask broadcasts across time. |
| Mask label values | Blank selects all positive labels; e.g. 1,2,5 selects only those labels. |
| Keep traces inside mask | Off restricts only seed positions. On also stops trajectories when they leave the selected mask. |
| Distribution | **Grid** (default) uses a regular physical lattice like autoflow; Random samples eligible voxel centers without replacement. |
| Grid interval unit | Voxel multiples preserve anisotropic resolution. Millimeters use the same physical interval along every axis. |
| Grid interval | Default **3 voxel widths**: pitches are 3*dx, 3*dy, 3*dz. The lattice is anchored to the input grid with a half-cell offset. Smaller intervals produce more seeds. |
| Cap Grid seed count | Off by default: use every eligible lattice point. Enable only when a separate count cap is desired. |
| Random seeds / 3D frame | Requested Random seeds per 3D frame, shown only in Random mode. The UI accepts up to 1,000,000; the integrator uses bounded batches. |
| 3D Grid seed cap | A separate count limit shown only when Grid and Cap Grid seed count are enabled. Hidden caps have no effect. |
| Random seed | Reproduces a Random seed selection; shown only for Random. |
| Seed minimum speed | Excludes low-speed seed candidates. Effective eligibility uses the greater of this value and the integration stop threshold. |

Grid seeds retain their regular pitch and are filtered by mask and magnitude.
A narrow region may contain no valid lattice point; reducing the Grid interval
increases coverage. The candidate lattice has the same 20-million-point guard
as autoflow; increasing the interval reduces that count. Random requested counts
cannot exceed available eligible voxel centers.

## Trace

| Parameter | Meaning |
| --- | --- |
| Direction | Downstream, upstream or both. Arrowheads always follow the input velocity. |
| Integrator | Midpoint (RK2) is faster; RK4 uses more field samples for better curved-field accuracy. |
| Step (mm) | Physical integration distance per step; smaller follows curves more closely and costs more computation. |
| Maximum steps / direction | Additional per-direction stopping limit; no algorithmic 1,000-step restriction. |
| Maximum length (mm) | Full trajectory length budget; bidirectional tracing splits it equally between both directions. |
| Minimum length (mm) | Discards complete seed trajectories shorter than this value; zero keeps short traces. |
| Stop-below speed | Stops integration in low-speed regions. This changes trajectories, unlike the display-only cutoff. |
| Maximum turn / step | Stops a step when the velocity direction turns more than this angle between its endpoints; 180 degrees disables it. |

Each trace uses a frozen frame; these are instantaneous streamlines, not
pathlines integrated through time. Integration also stops at grid/mask exits
and on return near a seed after a closed loop.

## Two distinct 2D modes

| Mode | Calculation | Speed used by thresholds and colors |
| --- | --- | --- |
| In-plane 2D streamlines | Project the velocity onto each slice's tangent basis, discard its normal component, and integrate that slice independently. It never follows trajectories into neighboring slices. | In-plane magnitude only; through-plane velocity cannot create a spurious planar trace. |
| 3D streamline slab projection | Compute actual 3D trajectories, clip their segments to a physical slab centered on each slice, and project the retained portions into the slice view. | Full 3D magnitude. |

**Random seeds / 2D slice** and **2D Grid cap / slice** are separate settings. Only the active one appears, and a Grid cap appears only when explicitly enabled. Both are independent of the 3D-frame counts. Grid interval and
mask constraints apply to both modes. **Slab thickness** applies only to 3D
projection; zero means one voxel along the slice's fixed axis.

Both modes precompute every spatial slice for each requested frame in the
background. Switching a mode can require its first calculation, after which its
saved result is reusable. Spatial navigation only retrieves the precomputed
slice geometry; it never starts integration or clipping. The 3D pane continues
to display full 3D trajectories independently of the chosen 2D algorithm.

## Display and cache

| Parameter | Meaning |
| --- | --- |
| Show 2D / Show 3D | Independent visibility toggles; the existing 3D pane Show toggle still gates rendering. |
| 2D arrowheads | Direction markers following the input flow. |
| Width / opacity | Streamline appearance only. Opacity is synchronized with the label-list Streamlines row. |
| Colormap | Flow (original palette), Turbo, Viridis, Plasma, Inferno, Magma, or Grayscale; the same palette is used in 2D and 3D without recomputation. |
| Ignore below speed | Hides cached segments below the cutoff in 2D and 3D without repeating or stopping integration. In-plane overlays use their in-plane magnitude. |
| Color maximum | Upper end of the color scale; higher values saturate at red and remain visible. |
| Recalculate | Bypasses saved results and recomputes the selected configuration. |
| Clear cache | Disables overlays and clears current geometry and saved streamline entries. |

Width, opacity, arrows, color and low-speed display filtering update immediately.
Integration changes are debounced and calculated off-thread. The current frame
is prioritized, followed by all other frames. Slab changes reuse 3D trace
caches; planar caches are independent of slab thickness and the 3D seed count.

Selected masks are snapshotted before worker access. Edits and undo/redo on the
selected mask invalidate stale work and request new cached results. Incompatible
component/mask grids are rejected; an incompatible active background hides the
overlay until a compatible grid is active again.

Completed frames stay in memory. Separate 3D trace, projection and planar NPZ
caches persist below Qt's per-user CacheLocation/streamlines-v1 directory. Keys
include actual loaded samples, mask selection, spatial metadata and computation
parameters. Files are atomically written without pickle; corrupt entries are
recomputed and older disk entries are pruned to a 2 GiB budget. Closing files
cancels work and releases current geometry while retaining reusable disk data.
Images and labels are never altered by streamline calculation.

## Display-only label-list entry

While streamline overlays are enabled, **Streamlines** appears alongside integer
labels. Its checkbox changes visibility; right-click the row to adjust opacity.
The row is not a paint target or a `LabelDefinition`, never writes voxel values,
and is not serialized in a saved label sequence. The panel opacity and row
opacity are synchronized. Effective line opacity multiplies that value by
**All labels opacity**; the 3D header opacity still affects label surfaces only.
Holding the label-hide key also temporarily hides 2D streamline overlays.

Changing visibility, opacity, colormap, color limits or display speed filtering
uses the existing geometry. See [performance and default-parameter evidence](streamline-performance-2026-10-26.md)
for measured speedups, numerical equivalence and the rationale/limits of defaults.
