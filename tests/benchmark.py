import pathlib
import time

import polars as ps
import tqdm

from batchmatch import PatternLibrary
from batchmatch.filters import *

DATA_ROOT = pathlib.Path(__file__).parent / "data"


class ProfileLibrary(PatternLibrary):

    def __init__(self, consolidate=True):
        smarts = ps.read_parquet(DATA_ROOT / "aizynth_smarts.parquet")["smarts"]
        super().__init__(smarts, consolidate=consolidate, filters=[])

        self.supported_filters = {
            "Ring": RingFilter(self.patterns),
            "Atom": AtomCountFilter(self.patterns),
            "RDKit": RDKitFilter(self.patterns),
            "Pattern": PatternFilter(self.patterns),
        }

    def set_filters(self, names):
        self.filters = [self.supported_filters[k] for k in names]

    def profile_matches(self, mols):
        times, prs = [], []

        for m in mols:
            start = time.time()
            if not self.filters:  # brute-force
                n = len(self.patterns)
                indices = range(n)
            else:
                indices = self.could_match(m)
                n = len(indices)

            out, hits = [], 0
            for i in indices:
                if m.HasSubstructMatch(self.patterns[i], useChirality=self.chiral):
                    out.extend(self.backmap[i])
                    hits += 1
            times.append(time.time() - start)
            prs.append(hits / n)

        return np.mean(times), np.mean(prs)


def benchmark():
    df = ps.read_parquet(DATA_ROOT / "zinc.parquet").sample(n=1000, seed=0)
    mols = [Chem.MolFromSmiles(m) for m in df["smiles"]]

    # Benchmark
    brutelib = ProfileLibrary(consolidate=False)
    lib = ProfileLibrary()

    combos = []
    for r in range(1, 5):
        combos.extend(itertools.combinations(lib.supported_filters.keys(), r=r))
    pbar = tqdm.tqdm("Benchmarking", total=(3 + len(combos)))

    schema = ["Strategy", "Match Time", "Precision"]
    metrics = [
        ["Naive (no consolidate)", *brutelib.profile_matches(mols)],
        ["Naive", *lib.profile_matches(mols)],
    ]
    pbar.update(2)

    # Try different filter combos
    for S in combos:
        best = [None, 1e10, None]
        for perm in itertools.permutations(S, r=len(S)):
            lib.set_filters(perm)
            out = [" > ".join(perm), *lib.profile_matches(mols)]
            if out[1] < best[1]:
                best = out
        metrics.append(best)
        pbar.update(1)

    # Simulate an ideal filter
    t = 0
    for i, m in enumerate(mols):
        for p in lib.patterns:
            start = time.time()
            if m.HasSubstructMatch(p, useChirality=True):
                t += time.time() - start
    metrics.append(["Ideal", t / len(mols), 1.0])
    pbar.update(1)

    # Format results
    summary = ps.DataFrame(metrics, schema=schema, orient="row")
    with ps.Config(tbl_rows=len(summary)) as cfg:
        cfg.set_tbl_formatting("ASCII_MARKDOWN")
        print(summary.sort(ps.col("Match Time"), descending=True))


if __name__ == "__main__":
    benchmark()
