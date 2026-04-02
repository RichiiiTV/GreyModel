from __future__ import annotations

import json
import sys

from .cli import cli_main
from .utils import json_default


def _should_print_result(argv: list[str], payload) -> bool:
    if payload is None:
        return False
    return not (argv and argv[0] == "ui" and "--dry-run" not in argv)


if __name__ == "__main__":
    argv = sys.argv[1:]
    result = cli_main(argv)
    if _should_print_result(argv, result):
        print(json.dumps(result, indent=2, sort_keys=True, default=json_default))
