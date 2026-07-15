# Changelog

All notable changes to SpatioTemporal Labeler are documented in this file.

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
