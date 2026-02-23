from pathlib import Path

import pytest


_CLEANUP_PATTERNS = (
    "array_test_*.png",
    "ArrayFractal_*.gif",
    ".deterministic_frame_*.png",
)


def _cleanup_generated_files() -> None:
    root = Path.cwd()
    for pattern in _CLEANUP_PATTERNS:
        for path in root.glob(pattern):
            path.unlink(missing_ok=True)


@pytest.fixture(scope="session", autouse=True)
def cleanup_generated_test_files():
    _cleanup_generated_files()
    yield
    _cleanup_generated_files()
