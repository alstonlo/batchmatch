import itertools
from collections import defaultdict

import networkx as nx
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem.rdFingerprintGenerator import GetRDKitFPGenerator

from batchmatch.smarts import parse_rdqatom


class Filter:

    def __call__(self, mol, indices):
        raise NotImplementedError


class RingFilter(Filter):

    def __init__(self, patterns):
        rings = defaultdict(list)
        has_ring = np.zeros([len(patterns)], dtype=bool)
        for i, p in enumerate(patterns):
            for atoms in self.enumerate_cycles(p):
                has_ring[i] = True
                R = self.isolate_ring(p, tuple(atoms))
                rings[R].append(i)
        self.rings = {k: np.asarray(L) for k, L in rings.items()}
        self.acyclic_mask = np.logical_not(has_ring)

        # Patterns don't have rings, nothing clever we can do
        self.trivial = np.all(self.acyclic_mask)

    @staticmethod
    def enumerate_cycles(mol):
        G = nx.Graph()
        G.add_edges_from([
            (b.GetBeginAtomIdx(), b.GetEndAtomIdx())
            for b in mol.GetBonds() if b.IsInRing()
        ])
        return nx.simple_cycles(G, length_bound=6)

    @staticmethod
    def isolate_ring(mol, atoms):
        ring = Chem.RWMol()

        backmap = []
        for i in atoms:
            a = mol.GetAtomWithIdx(i)
            r = Chem.Atom(a.GetSymbol())
            backmap.append(ring.AddAtom(r))
        for src in range(len(atoms)):
            dst = (src + 1) % len(atoms)
            btype = mol.GetBondBetweenAtoms(atoms[src], atoms[dst]).GetBondType()
            if btype == Chem.BondType.UNSPECIFIED:
                raise ValueError("unspecified bond is unsupported")
            ring.AddBond(backmap[src], backmap[dst], order=btype)

        return Chem.MolToSmiles(ring)

    def __call__(self, mol, indices):
        if self.trivial:
            return indices

        # Eliminate based on existing ring systems
        mask = self.acyclic_mask.copy()
        for atoms in self.enumerate_cycles(mol):
            R = self.isolate_ring(mol, tuple(atoms))
            if R in self.rings:
                mask[self.rings[R]] = True
        if len(indices) < len(mask):
            mask = mask[indices]
        return indices[mask]


class AtomCountFilter(Filter):

    def __init__(self, patterns):
        bags = []
        for p in patterns:
            B = [parse_rdqatom(a) for a in p.GetAtoms()]
            bags.append(B)
        vocab = set(itertools.chain(*bags))

        self.vocab = list(vocab)
        self.vocab_qas = [q.rdmol for q in self.vocab]

        # Atom occurrence matrix
        X = []
        backmap = {qa: i for i, qa in enumerate(self.vocab)}
        for B in bags:
            counts = np.zeros([len(vocab)], dtype=np.uint8)
            for q in B:
                counts[backmap[q]] += 1
            X.append(counts)
        self.X = np.stack(X, axis=0)

    def __call__(self, mol, indices):
        if mol.GetNumAtoms() >= 256:
            raise ValueError("too many atoms")
        fp = [len(mol.GetAtomsMatchingQuery(qa)) for qa in self.vocab_qas]
        fp = np.asarray(fp, dtype=self.X.dtype)
        mask = np.all(fp >= self.X[indices, :], axis=-1)
        return indices[mask]


class BitFingerprintFilter(Filter):

    def __init__(self, patterns):
        self.X = np.stack([self.fingerprint(p) for p in patterns], axis=0)

    def fingerprint(self, mol):
        raise NotImplementedError()

    def __call__(self, mol, indices):
        q = self.fingerprint(mol)
        X = self.X[indices, :]
        mask = ~np.any((~q) & X, axis=-1)
        return indices[mask]


class RDKitFilter(BitFingerprintFilter):

    def __init__(self, patterns, bits=2048, L=4):
        self.fac = GetRDKitFPGenerator(fpSize=bits, minPath=2, maxPath=L, useHs=False)

        super().__init__(patterns)

    def fingerprint(self, mol):
        invariants = [a.GetAtomicNum() for a in mol.GetAtoms()]
        fp = self.fac.GetFingerprintAsNumPy(mol, customAtomInvariants=invariants)
        return np.packbits(fp)


# Reference:
#   https://github.com/ml-jku/mhn-react/blob/main/mhnreact/retrosyn.py
# Optimized a bit with packing (pun intended)
class PatternFilter(BitFingerprintFilter):

    def __init__(self, patterns, bits=2048):
        self.bits = bits

        super().__init__(patterns)

    def fingerprint(self, mol):
        fp = Chem.PatternFingerprint(mol, fpSize=self.bits)
        array = np.zeros((0,), dtype=bool)
        DataStructs.ConvertToNumpyArray(fp, array)
        return np.packbits(array)
