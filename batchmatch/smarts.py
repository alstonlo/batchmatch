import dataclasses
import re
from typing import Optional

from rdkit import Chem

QUERY_FIELDS = {  # Map RDKit -> AtomQuery
    "Type": None,
    "AtomicNum": "number",
    "IsAromatic": "aromatic",
    "ExplicitDegree": "degree",
    "HCount": "Hs",
    "FormalCharge": "charge",
}

QUERY_REGEX = "|".join([f"Atom{k}" for k in QUERY_FIELDS])
QUERY_REGEX = f"^(AtomAnd|(({QUERY_REGEX}) (-?\d+) = val))$"
QUERY_REGEX = re.compile(QUERY_REGEX)


@dataclasses.dataclass(frozen=True, order=True)
class AtomQuery:

    number: int
    aromatic: Optional[bool] = None
    degree: Optional[int] = None
    Hs: Optional[int] = None
    charge: Optional[int] = None

    @property
    def rdmol(self):
        query = [f"#{self.number}"]
        if self.aromatic is not None:
            query.append("a" if self.aromatic else "A")
        if self.degree is not None:
            query.append(f"D{self.degree}")
        if self.Hs is not None:
            query.append(f"H{self.Hs}")
        if self.charge is not None:
            query.append(f"{self.charge:+}")
        query = "[" + "&".join(query) + "]"
        return Chem.MolFromSmarts(query).GetAtomWithIdx(0)


def parse_rdqatom(atom: Chem.QueryAtom) -> AtomQuery:
    query = str(atom.DescribeQuery()).strip()
    kwargs = dict()

    for expr in query.split("\n"):
        expr = expr.strip()
        if not QUERY_REGEX.match(expr):
            raise ValueError(f"Unsupported query atom:\n{query}")
        if expr == "AtomAnd":
            continue
        prop, value, *_ = expr.split(" ")  # e.g., 'AtomAtomicNum 6 = val'
        prop = prop.replace("Atom", "", 1)

        if prop == "Type":
            kwargs["number"] = atom.GetAtomicNum()
            kwargs["aromatic"] = atom.GetIsAromatic()
        else:
            dtype = bool if (prop == "IsAromatic") else int
            kwargs[QUERY_FIELDS[prop]] = dtype(value)

    return AtomQuery(**kwargs)
