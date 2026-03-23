"""Bootstrap package for running GreyModel directly from a source checkout.

This repo uses a ``src/`` layout for packaging. A fresh clone that has not been
installed yet would otherwise fail on ``python -m greymodel`` because the
package directory lives under ``src/greymodel``. This bootstrap package points
imports at the real source package so the checkout is directly runnable.
"""

from __future__ import annotations

from pathlib import Path

_SOURCE_PACKAGE_DIR = Path(__file__).resolve().parent.parent / "src" / "greymodel"
_SOURCE_INIT = _SOURCE_PACKAGE_DIR / "__init__.py"

if not _SOURCE_INIT.exists():
    raise ModuleNotFoundError(f"GreyModel source package not found at {_SOURCE_INIT}")

__path__ = [str(_SOURCE_PACKAGE_DIR)]
__file__ = str(_SOURCE_INIT)

if __spec__ is not None:
    __spec__.submodule_search_locations = __path__

with _SOURCE_INIT.open("rb") as source_handle:
    exec(compile(source_handle.read(), __file__, "exec"))
