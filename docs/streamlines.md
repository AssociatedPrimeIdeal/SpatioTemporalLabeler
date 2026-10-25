# Streamline overlays

Load the three flow component image sequences along with the image you want to
view. Open **View > Streamlines** (or the **Streamlines** toolbar button) and
check **Enable streamline overlays**. Each input can be selected independently
from the loaded image sequences.

The Segdata preparation convention is:

| Default input | Component convention | Positive anatomical direction |
| --- | --- | --- |
| `flow_x` | HF | F, inferior |
| `flow_y` | AP | P, posterior |
| `flow_z` | RL | L, left |

Change the anatomical direction beside each input when using other conventions.
The three components must cover R/L, A/P and H/F once each. Their voxel grids
must match. Frame counts must agree, except that a 3D input can broadcast across
the frames of a 4D input. Image display flips do not change vector directions.

## Input units

The default **Source values (unscaled)** mode preserves the input numbers and
labels thresholds and colors in source units. Existing preparation outputs mix
phase in radians with calibrated velocity, so a `flow_*` filename alone does
not establish its units.

- For velocity, select **cm/s** or **m/s**.
- For phase, select **Phase (rad) + VENC** and enter the VENC in cm/s for each
  selected input. Calibration applies `phase * VENC / pi` before mapping vectors
  into anatomical space. All three VENC values must be positive.

This module follows the supplied field; it does not unwrap aliased phase.

## Live controls and navigation

The module integrates a frozen field at each time frame in both directions.
These are instantaneous streamlines, not time-integrated pathlines. Seeds are
deterministic samples of finite flow above the selected minimum magnitude
throughout the input field. Step size and the full-trajectory length limit use
physical millimeters.

The X-Y, X-Z and Y-Z overlays show the 3D traces projected through a slab around
the current slice. A slab setting of zero means one voxel along that view's
fixed axis. Direction arrowheads follow the supplied flow. 2D and 3D visibility
can be controlled independently; the 3D pane's existing opacity slider can make
label surfaces translucent so that interior streamlines are visible.

Line width, opacity, arrows and the color scale update immediately. Integration
changes are debounced and computed in the background. Changing the slab reuses
cached traces and rebuilds only the projections. The current frame is computed
first, followed by all other frames. Once a frame is ready, every spatial slice
has a cached projection: slice navigation never starts integration or clipping.
An incompatible background image hides the overlay until a compatible grid and
time axis are active again.

## Cache behavior

Completed frames stay in memory. Trace and projection results also persist as
separate, atomically written NPZ files under Qt's per-user cache directory,
`streamlines-v1`. Cache keys include actual loaded sample values, voxel geometry,
component mapping, calibration and computation settings. Corrupt entries are
recomputed, and older disk entries are pruned to a 2 GiB budget.

**Recalculate** bypasses saved entries. **Clear cache** disables overlays and
clears memory and saved streamline results. Closing files cancels outstanding
work and releases current geometry while keeping reusable disk entries. The
module does not modify source images or labels.
