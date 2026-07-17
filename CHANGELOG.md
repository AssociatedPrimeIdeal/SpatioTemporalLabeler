# Changelog

All notable changes to SpatioTemporal Labeler are documented in this file.

## 0.2.2 - 2026-07-17

### Changed

- Update application for 0.2.2

## 0.2.1 - 2026-07-17

### Changed

- Update application for 0.2.1

## 0.2.0 - 2026-07-17

### Added

- Immediate scissors-lasso editing in 2D slices and the 3D view, with erase and label-replacement operations, implicit dashed closure, automatic overlay clearing, and all-time-frame support.
- Per-label opacity controls in label-row context menus and a persistent global label opacity control.

### Changed

- Threshold candidates are now materialized with Apply as a single replaceable `Threshold mask` entry whose checkbox and deletion control the editing constraint.
- 3D camera rotation now uses Alt+left drag so unmodified left drag is reserved for scissors-lasso editing.

## 0.1.6 - 2026-07-17

### Added

- Draggable triangular handles at both visible ends of each spatial locator line for fast linked X/Y/Z slice navigation.

### Fixed

- 3D camera interaction now keeps a padded clipping range around every visible label segment, preventing rotated surfaces from disappearing or exposing clipped interiors.

## 0.1.5 - 2026-07-16

### Added

- Selectable X-Y/X-Z/Y-Z/X-T/Y-T/Z-T planes for other-image previews, with automatic plane following during Shift-location.
- Persistent 3D rendering controls for clinical, matte, and glossy styles, lighting, surface smoothing, and detail level.

### Changed

- Middle-button dragging now adjusts window width horizontally and window level vertically; Shift+left drag remains the 2D pan gesture.
- Threshold lower and upper controls now use percentages mapped into the image's real intensity range.
- Time navigation coalesces complete visible-label 3D updates to the latest requested frame with adaptive throttling.

### Performance

- Window level/width changes update only grayscale levels instead of rebuilding slices and overlays.
- 3D cursor rendering is coalesced, threshold previews use per-frame caches, and spatial threshold operations calculate only affected frames.

## 0.1.0 - 2026-07-15

### Added

- 3D/4D NIfTI and NRRD loading with metadata-preserving save and drag-and-drop classification.
- Linked X-Y, X-Z, Y-Z, and selectable X-T/Y-T/Z-T editing views.
- Multilabel 3D surface rendering with progressive time-slider updates.
- Brush, eraser, closed-contour fill, physical footprints, and linked positioning.
- Sparse undo/redo and optional all-time-frame spatial editing.
- Multiple image and label sequences, collapsible previews, and unified import classification.
- Per-label morphology for small-component removal, hole filling, opening, and closing.
- Physical-unit morphology parameters: component volumes use `mm³`, and opening/closing radii use `mm` with anisotropic spacing.
- Signed-distance label interpolation between user-selected keyframes, recorded as one undoable edit.
- 3D label mapping onto 4D images, with all-frame replication by default or placement in one selected frame.
- Threshold masks with live preview, manual and automatic methods, and a held bypass shortcut.
- Window level/width controls, label picking, and 2D/3D seed region growing.
- 2D zoom, pan, reset, full-panel maximize, and per-axis locator colors.
- Middle-button panning in every editable 2D view.
- Continuous brush and eraser gestures now update all four visible 2D label overlays in real time while rebuilding 3D surfaces only after release.
- The 3D camera now uses timer-free trackball interaction to prevent unintended continuous rotation.
- Each label now renders through an independent padded, closed, smoothed, decimated surface actor.
- Persistent label metadata for NRRD and NIfTI.
- A film-strip application icon for Linux and Windows packages.
- English and Simplified Chinese interfaces.
- Linux, Windows, wheel, and source-distribution release automation.
