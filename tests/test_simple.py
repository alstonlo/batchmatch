import pathlib

import polars as ps
import pytest
from rdkit import Chem

from batchmatch import PatternLibrary
from batchmatch.filters import AtomCountFilter, PatternFilter, RDKitFilter, RingFilter

DATA_ROOT = pathlib.Path(__file__).parent / "data"

SUPPORTED_FILTERS = [
    RingFilter,
    AtomCountFilter,
    RDKitFilter,
    PatternFilter,
]

HARD_SMILES = [
    "CC[C@@]12CC[C@@](C(=O)N(C)C[C@@H](C)NC(=O)c3cccc(C(C)(F)F)n3)(CO1)C2"  # fused ring
]


@pytest.fixture()
def aizynth_smarts():
    df = ps.read_parquet(DATA_ROOT / "aizynth_smarts.parquet")
    return df["smarts"]


@pytest.fixture()
def aizynth_patterns(aizynth_smarts):
    return PatternLibrary(aizynth_smarts, consolidate=False, filters=[])


@pytest.fixture()
def random_aizynth_tasks(aizynth_patterns):
    df = ps.read_parquet(DATA_ROOT / "zinc.parquet").sample(n=20)
    mols = [Chem.MolFromSmiles(smi) for smi in df["smiles"].to_list() + HARD_SMILES]
    return [(m, set(aizynth_patterns.match(m))) for m in mols]


@pytest.mark.parametrize("filter_class", SUPPORTED_FILTERS)
def test_filter(filter_class, aizynth_patterns, random_aizynth_tasks):
    filt = filter_class(aizynth_patterns.patterns)
    for m, matches in random_aizynth_tasks:
        I = filt(m, aizynth_patterns.universe).tolist()
        assert matches <= set(I), Chem.MolToSmiles(m)


def test_consolidation(aizynth_smarts, random_aizynth_tasks):
    patterns = PatternLibrary(aizynth_smarts, filters=[])
    for m, matches in random_aizynth_tasks:
        assert set(patterns.match(m)) == matches


def test_default(aizynth_smarts, random_aizynth_tasks):
    patterns = PatternLibrary(aizynth_smarts)
    for m, matches in random_aizynth_tasks:
        assert set(patterns.match(m)) == matches
