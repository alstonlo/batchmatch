from typing import Callable, List, Sequence

import numpy as np
from rdkit import Chem, rdBase

from batchmatch.filters import AtomCountFilter, Filter, RDKitFilter, RingFilter

DEFAULT_FILTERS = (RingFilter, AtomCountFilter, RDKitFilter)


class PatternLibrary:

    def __init__(
        self,
        smarts: Sequence[str],
        filters: Sequence[Callable[[Sequence], Filter]] = DEFAULT_FILTERS,
        chiral: bool = True,
        consolidate: bool = True,
    ):
        self.smarts = tuple(smarts)

        # Do a bit of SMARTS cleanup
        patterns = []
        for i, sma in enumerate(smarts):
            p = Chem.MolFromSmarts(sma)
            for a in p.GetAtoms():
                a.SetAtomMapNum(0)  # for safety
            with rdBase.BlockLogs():
                p.UpdatePropertyCache(strict=False)
            patterns.append(p)

        # Since input might have duplicates, (e.g., if extracted from a set of reactions)
        # we'll try to extract a unique set of templates.
        if consolidate:
            self.patterns, self.backmap = self.consolidate(patterns)
        else:
            self.patterns = patterns
            self.backmap = tuple([i] for i in range(len(self.patterns)))

        self.filters = [fac(self.patterns) for fac in filters]
        self.chiral = chiral

        # Initialize this once
        self.universe = np.arange(len(self.patterns), dtype=np.int32)

    @staticmethod
    def consolidate(patterns):
        buckets = dict()
        for i, p in enumerate(patterns):
            # TODO: canonicalize SMARTS with rdcanon for better consolidation
            sma = Chem.MolToSmarts(p)
            if sma not in buckets:
                buckets[sma] = (p, [i])
            else:
                buckets[sma][1].append(i)
        return zip(*buckets.values())

    def could_match(self, mol: Chem.Mol) -> List[int]:
        indices = self.universe
        for f in self.filters:
            indices = f(mol, indices)
        return indices.tolist()

    def match(self, mol: Chem.Mol) -> List[int]:
        if not self.filters:  # brute-force
            indices = self.universe
        else:
            indices = self.could_match(mol)

        out = []
        for i in indices:
            if mol.HasSubstructMatch(self.patterns[i], useChirality=self.chiral):
                out.extend(self.backmap[i])
        return out
