import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def model_path():
    """A real .cact model: $SPINDLE_TEST_MODEL or the first file in models/."""
    path = os.environ.get('SPINDLE_TEST_MODEL') or next(iter(sorted((ROOT / 'models').glob('*.cact'))), None)
    if not path:
        pytest.skip('No model: set SPINDLE_TEST_MODEL or put a .cact file in models/')
    return Path(path)
