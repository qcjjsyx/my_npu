"""Allow ``python -m mac_experiment``."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
