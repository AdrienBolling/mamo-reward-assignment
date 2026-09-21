"""The package skeleton imports, and every layer states what belongs in it."""

import importlib

import pytest

LAYERS = [
    "mamora",
    "mamora.agents",
    "mamora.analysis",
    "mamora.connectors",
    "mamora.envs",
    "mamora.operators",
    "mamora.runner",
]


@pytest.mark.parametrize("name", LAYERS)
def test_layer_imports_and_documents_its_role(name):
    module = importlib.import_module(name)
    assert module.__doc__, f"{name} must state what belongs in it"
