from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from simulation.simulation.run_simulation import main


def _force_webshop_domain() -> None:
    for index, argument in enumerate(sys.argv[1:], start=1):
        if argument.startswith("--domain="):
            if argument.split("=", 1)[1] != "webshop":
                raise SystemExit("WebShop entrypoint does not accept another domain.")
            return
        if argument == "--domain":
            if index + 1 >= len(sys.argv) or sys.argv[index + 1] != "webshop":
                raise SystemExit("WebShop entrypoint requires --domain webshop.")
            return
    sys.argv.extend(["--domain", "webshop"])


if __name__ == "__main__":
    _force_webshop_domain()
    main()
