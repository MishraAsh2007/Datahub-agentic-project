"""Windows workaround for `datahub datapack load`.

acryl-datahub 1.7.0 resolves a filesystem backend by URL-parsing the path:

    datahub/ingestion/fs/fs_base.py
    def get_path_schema(path):
        scheme = parse.urlparse(path).scheme
        if scheme == "":
            scheme = "file"
        return scheme

On Windows an absolute path like C:\\Users\\... parses with scheme "c" (the
drive letter), so the "file" fallback never fires and fs_registry.get("c")
raises KeyError: 'Did not find a registered class for c'.

The datapack loader always feeds absolute cache paths, so `datahub datapack
load` cannot succeed on Windows without this patch.

Usage mirrors the real CLI:
    python datapack_load_win.py load showcase-ecommerce
    python datapack_load_win.py list
"""

import re
import sys
from urllib import parse

# A leading single-letter scheme is a Windows drive letter, not a URL scheme.
_WINDOWS_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")


def _patched_get_path_schema(path: str) -> str:
    if _WINDOWS_DRIVE.match(path):
        return "file"
    scheme = parse.urlparse(path).scheme
    if scheme == "":
        scheme = "file"
    return scheme


def _apply_patch() -> None:
    from datahub.ingestion.fs import fs_base

    fs_base.get_path_schema = _patched_get_path_schema

    # file.py did `from ...fs_base import get_path_schema`, binding the name in
    # its own namespace, so patching fs_base alone would not reach it.
    from datahub.ingestion.source import file as file_source

    file_source.get_path_schema = _patched_get_path_schema


if __name__ == "__main__":
    _apply_patch()

    from datahub.cli.datapack.datapack_cli import datapack

    sys.exit(datapack(sys.argv[1:], standalone_mode=False))
