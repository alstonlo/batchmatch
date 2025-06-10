<div align="center">

# batchmatch

</div>

<br>

---

This package accelerates the matching of a query molecule to a large batch of patterns, given as SMARTS strings.
This is a common subroutine in template-based retrosynthesis, where one is interested in which of 10-100k reaction
templates are applicable for any given molecule. The key insight is that we can first use cheap heuristics
to rule out patterns that can never match to avoid solving an expensive subgraph isomorphism problem.

## Installation

Install `batchmatch` using pip:

```bash
pip install batchmatch 
```

## Usage

The `batchmatch` library has one core object `PatternLibrary`, which takes as input a list of SMARTS strings and allows
users to quickly query molecules against them using its `match()` method:

```python
import batchmatch
from rdkit import Chem

with open("smarts.txt", "r") as f:
    smarts = f.read().splitlines()
query = Chem.MolFromSmiles("Cn1c(=O)c2c(ncn2C)n(C)c1=O")

# Instantiate the pattern library 
# Importantly, this only has to be done once, as it computes some indices  
patterns = batchmatch.PatternLibrary(
    smarts=smarts,
    consolidate=True,  # default; whether to deduplicate SMARTS 
    filters=batchmatch.DEFAULT_FILTERS,  # default; the filter heuristics used 
    chiral=True,  # default; whether to use chirality when matching
)

# Match the query molecule
for i in patterns.match(query):
    print(f"Query matches {smarts[i]}")
```

### How does it work?

Under the hood, `PatternLibrary` first consolidates (deduplicates) the input SMARTS. Then, it applies a series of
heuristics (filters) to rule out blatantly unmatching patterns, which can be run in isolation with the `could_match()`
method. Currently, `batchmatch` supports 4 filters:

1. `RingFilter`: rules out patterns with rings not present in the query.
2. `AtomCountFilter`: rules out patterns based on their atom type counts. For example, a pattern with three oxygens can
   never match a molecule with only one.
3. `RDKitFilter`: uses a modified version of the RDKit fingerprint. Specifically, we rule out a pattern if their
   fingerprint's on bits are not a subset of the query's.
4. `PatternFilter`: the same as above but using RDKit's `PatternFingerprint`, which was adapted
   from [MHN-React](https://github.com/ml-jku/mhn-react).

Empirically, we find the Ring > Atom > RDKit filter combination (in that order) works best and is what we use by
default. Note that the bulk of the computation used in the filters (namely, computing statistics about the patterns) is
done and cached on instantiation of `PatternLibrary`. Finally, we run an exhaustive substructure match with the
remaining patterns to yield the final results.

### Limitations

While `batchmatch` is well-tested on templates from AIZynthFinder and queries from ChEMBL, it is difficult to
anticipate every use case due to the expansiveness of SMARTS. The strategies used herein are highly tailored towards
SMARTS extracted from reaction templates, which only use a narrow subset of the specification language.
I would always recommend running some spot checks and benchmarks on your data to see if `batchmatch` works well. For
convenience,

```python
batchmatch.PatternLibrary(smarts=smarts, consolidate=False, filters=[])
```

performs the brute-force strategy of matching against every SMARTS with no extra tricks!

## Benchmarks

Below, I have run an informal benchmarks (`test/benchmark.py`) on my laptop (Intel Core i7-12800H processor),
using the AIZynthFinder template set and 1k random molecules from ZINC. For each combination of 0-4 filters, I report
the _best_ permutation below in terms of match time (s) per molecule. I also consider the precision, the number of true
matches divided by number of exhaustive substructure matches. An ideal filter (**Ideal**) would filter out exactly the
true matches and therefore have a precision of 1, but _not_ a match time of 0 because one still has to perform the
matching routine.

<center>

| Strategy                      | Match Time | Precision |
|-------------------------------|------------|-----------|
| Naive (no consolidate)        | 0.224459   | -         |
| Naive                         | 0.146469   | 0.002347  |
| Ring                          | 0.094124   | 0.004005  |
| Atom                          | 0.029731   | 0.022558  |
| Ring > Atom                   | 0.023717   | 0.029551  |
| Pattern                       | 0.01542    | 0.076733  |
| Ring > Pattern                | 0.013881   | 0.076841  |
| RDKit                         | 0.013821   | 0.102975  |
| RDKit > Pattern               | 0.013694   | 0.122205  |
| Ring > RDKit > Pattern        | 0.012047   | 0.122327  |
| Ring > RDKit                  | 0.011598   | 0.118188  |
| Ring > Atom > Pattern         | 0.008725   | 0.143111  |
| Atom > Pattern                | 0.008391   | 0.142975  |
| Ring > Atom > RDKit > Pattern | 0.007981   | 0.210596  |
| Atom > RDKit > Pattern        | 0.007611   | 0.210462  |
| Atom > RDKit                  | 0.00745    | 0.181535  |
| Ring > Atom > RDKit           | 0.007396   | 0.20821   |
| Ideal                         | 0.000496   | 1.0       |

</center>

Overall, we find that our default setting is almost 20x faster than a naive implementation and over 80x as precise!
Compared to the **Pattern** strategy in MHN-React, we are 2x faster. However, there is still a lot of potential for
improvement with a 15x slowdown and 5x decreased precision compared to the theoretical optimum. Likely, porting our
filters to C++ will yield easy speed improvements, though I lack the knowledge to do this currently.