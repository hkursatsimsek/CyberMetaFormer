"""Ortak fixture'lar. <repo> içinden çalıştırılır:

    venv_cyborg/bin/python -m pytest cybermetaformer/hidden_state/tests -q
"""
import numpy as np
import pytest

from minicage.dataset import loaders as L
from cybermetaformer.hidden_state import data as D


@pytest.fixture(scope='session')
def store():
    return D.EpisodeStore()


@pytest.fixture(scope='session')
def raw():
    raw, _ = L.load_raw(D.DEFAULT_DATA, fields=['obs_blue', 'true_state', 'blue_policy', 'red_policy'])
    return raw


@pytest.fixture(scope='session')
def splits():
    return L.load_splits(D.DEFAULT_DATA)


@pytest.fixture(scope='session')
def small_idx(splits):
    rng = np.random.default_rng(1)
    return np.sort(rng.choice(splits['train'], 40, replace=False))
