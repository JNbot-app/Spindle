import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def model_path():
    """A real .cact model: $SPINDLE_TEST_MODEL or models/spindle.cact."""
    path = os.environ.get('SPINDLE_TEST_MODEL') or ROOT / 'models' / 'spindle.cact'
    if not Path(path).exists():
        pytest.skip('No model: set SPINDLE_TEST_MODEL or download spindle.cact into models/')
    return Path(path)
