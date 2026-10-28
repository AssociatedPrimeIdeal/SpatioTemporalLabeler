from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

if __name__ == "__main__":
    from multiprocessing import freeze_support

    freeze_support()
    from spatiotemporal_labeler.app import main

    raise SystemExit(main())
