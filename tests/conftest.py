"""Shared pytest configuration and fixtures."""
import sys
import os

# Ensure project root is on sys.path so modules can be imported without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import state


@pytest.fixture(autouse=False)
def clean_state():
    """Reset all game state before (and after) a test."""
    state.reset()
    yield
    state.reset()
