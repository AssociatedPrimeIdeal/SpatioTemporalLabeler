# Sample 3D+t Data

This directory contains one de-identified 18-frame PCMRA image sequence and its label sequence:

- `pcmra.seq.nrrd`: scalar image, source shape `(110, 112, 74, 18)`
- `seg.seq.nrrd`: integer labels, source shape `(110, 112, 74, 18)`

Both files use a time-last NRRD layout and an approximately 2.4 mm isotropic spatial grid. They normalize to canonical `[X,Y,Z,T]` shape `(74, 112, 110, 18)` in the application.

Run the sample from the repository root:

```bash
conda activate ryy
python main.py examples/sample-data
```

The sample is intended for application testing and demonstration.
