from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from simulation.simulation.run_simulation import main


if __name__ == "__main__":
    if "--domain" not in sys.argv:
        sys.argv.extend(["--domain", "travelplanner"])
    main()
