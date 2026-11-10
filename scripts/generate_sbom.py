"""Generate a minimal CycloneDX-like inventory from installed distributions."""
from __future__ import annotations

import json
from importlib.metadata import distributions


def main() -> None:
    components = sorted(
        ({"name": dist.metadata["Name"], "version": dist.version, "type": "library"}
         for dist in distributions() if dist.metadata.get("Name")),
        key=lambda component: component["name"].lower(),
    )
    print(json.dumps({"bomFormat": "CycloneDX", "specVersion": "1.5", "components": components}, indent=2))


if __name__ == "__main__":
    main()
