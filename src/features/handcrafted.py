"""
Hand-crafted sequence feature extractors.

Implemented features:
  - AAC   : Amino Acid Composition (20-dim)
  - DPC   : Dipeptide Composition (400-dim)
  - PAAC  : Pseudo-Amino Acid Composition Type-1 (20+lambda dim)
  - GAAC  : Grouped Amino Acid Composition (5-dim)
  - CTDC  : Composition based on physicochemical groups (21-dim)
  - CTDT  : Transition based on physicochemical groups (21-dim)
  - CTDD  : Distribution based on physicochemical groups (105-dim)
  - QSO   : Quasi-Sequence Order (20+2*maxlag dim)

All extractors follow the same interface:
    extract(sequence: str) -> np.ndarray
    extract_batch(sequences: List[str]) -> np.ndarray
    feature_names() -> List[str]
"""
from itertools import product
from typing import List

import numpy as np
import pandas as pd

# ─── constants ─────────────────────────────────────────────────────────────

AAS = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {aa: i for i, aa in enumerate(AAS)}

# Physicochemical property values for PAAC (Tanford / Hopp-Woods / mass)
# Source: Chou & Shen, 2007 — normalised to zero-mean unit-variance
_H1 = {"A":  0.620, "C":  0.290, "D": -0.900, "E": -0.740, "F":  1.190,
       "G":  0.480, "H": -0.400, "I":  1.380, "K": -1.500, "L":  1.060,
       "M":  0.640, "N": -0.780, "P":  0.120, "Q": -0.850, "R": -2.530,
       "S": -0.180, "T": -0.050, "V":  1.080, "W":  0.810, "Y":  0.260}

_H2 = {"A": -0.500, "C": -1.000, "D":  3.000, "E":  3.000, "F": -2.500,
       "G":  0.000, "H": -0.500, "I": -1.800, "K":  3.000, "L": -1.800,
       "M": -1.300, "N":  0.200, "P":  0.000, "Q":  0.200, "R":  3.000,
       "S":  0.300, "T": -0.400, "V": -1.500, "W": -3.400, "Y": -2.300}

_MASS = {"A": 15.0, "C": 47.0, "D": 59.0, "E": 73.0, "F": 91.0,
         "G":  1.0, "H": 82.0, "I": 57.0, "K": 73.0, "L": 57.0,
         "M": 75.0, "N": 58.0, "P": 42.0, "Q": 72.0, "R": 101.0,
         "S": 31.0, "T": 45.0, "V": 43.0, "W": 130.0, "Y": 107.0}


def _normalise(prop: dict) -> dict:
    vals = np.array(list(prop.values()))
    mu, sigma = vals.mean(), vals.std()
    return {k: (v - mu) / (sigma + 1e-8) for k, v in prop.items()}


H1 = _normalise(_H1)
H2 = _normalise(_H2)
MS = _normalise(_MASS)

PAAC_PROPS = [H1, H2, MS]  # 3 physicochemical properties

# Grouped amino acids (5 groups)
GAAC_GROUPS = {
    "Aliphatic":  set("GAVLMI"),
    "Aromatic":   set("FYWH"),
    "Positive":   set("KRH"),
    "Negative":   set("DE"),
    "Uncharged":  set("STCPNQ"),
}
GAAC_ORDER = ["Aliphatic", "Aromatic", "Positive", "Negative", "Uncharged"]

# CTDC/CTDT/CTDD physicochemical groups
# 7 properties, each with 3 classes
CTDC_PROPS = {
    "hydrophobicity": {
        1: set("RKEDQN"),
        2: set("GASTPHY"),
        3: set("CVLIMFW")
    },
    "normalized_vdw": {
        1: set("GASTCPD"),
        2: set("NVEQIL"),
        3: set("MHKFRYW")
    },
    "polarity": {
        1: set("LIFWCMVY"),
        2: set("PATGS"),
        3: set("HQRKNED")
    },
    "polarizability": {
        1: set("GASDT"),
        2: set("CPNVEQIL"),
        3: set("KMHFRYW")
    },
    "charge": {
        1: set("KR"),
        2: set("ANCQGHILMFPSTWYV"),
        3: set("DE")
    },
    "secondary_structure": {
        1: set("EALMQKRH"),
        2: set("VIYCWFT"),
        3: set("GNPSD")
    },
    "solvent_accessibility": {
        1: set("ALFCGIVW"),
        2: set("RKQEND"),
        3: set("MPSTHY")
    },
}

# Schneider-Wrede & Grantham distance matrices for QSO
# Values are precomputed pairwise distances between the 20 AAs
_SW_MATRIX = {
    ("A","A"):0.000,("A","C"):0.114,("A","D"):0.111,("A","E"):0.074,("A","F"):0.258,
    ("A","G"):0.079,("A","H"):0.142,("A","I"):0.195,("A","K"):0.105,("A","L"):0.203,
    ("A","M"):0.207,("A","N"):0.114,("A","P"):0.179,("A","Q"):0.101,("A","R"):0.142,
    ("A","S"):0.062,("A","T"):0.120,("A","V"):0.149,("A","W"):0.333,("A","Y"):0.243,
    ("C","C"):0.000,("C","D"):0.174,("C","E"):0.154,("C","F"):0.224,("C","G"):0.152,
    ("C","H"):0.175,("C","I"):0.176,("C","K"):0.183,("C","L"):0.184,("C","M"):0.157,
    ("C","N"):0.138,("C","P"):0.166,("C","Q"):0.155,("C","R"):0.182,("C","S"):0.113,
    ("C","T"):0.174,("C","V"):0.145,("C","W"):0.302,("C","Y"):0.209,("D","D"):0.000,
    ("D","E"):0.073,("D","F"):0.300,("D","G"):0.118,("D","H"):0.115,("D","I"):0.244,
    ("D","K"):0.120,("D","L"):0.252,("D","M"):0.237,("D","N"):0.064,("D","P"):0.187,
    ("D","Q"):0.090,("D","R"):0.162,("D","S"):0.096,("D","T"):0.109,("D","V"):0.209,
    ("D","W"):0.361,("D","Y"):0.266,("E","E"):0.000,("E","F"):0.268,("E","G"):0.116,
    ("E","H"):0.090,("E","I"):0.222,("E","K"):0.060,("E","L"):0.230,("E","M"):0.211,
    ("E","N"):0.082,("E","P"):0.171,("E","Q"):0.052,("E","R"):0.107,("E","S"):0.093,
    ("E","T"):0.097,("E","V"):0.187,("E","W"):0.336,("E","Y"):0.240,("F","F"):0.000,
    ("F","G"):0.296,("F","H"):0.163,("F","I"):0.101,("F","K"):0.268,("F","L"):0.107,
    ("F","M"):0.114,("F","N"):0.215,("F","P"):0.195,("F","Q"):0.199,("F","R"):0.243,
    ("F","S"):0.233,("F","T"):0.210,("F","V"):0.137,("F","W"):0.155,("F","Y"):0.094,
    ("G","G"):0.000,("G","H"):0.161,("G","I"):0.236,("G","K"):0.138,("G","L"):0.244,
    ("G","M"):0.251,("G","N"):0.114,("G","P"):0.124,("G","Q"):0.124,("G","R"):0.177,
    ("G","S"):0.056,("G","T"):0.142,("G","V"):0.196,("G","W"):0.369,("G","Y"):0.272,
    ("H","H"):0.000,("H","I"):0.184,("H","K"):0.086,("H","L"):0.193,("H","M"):0.167,
    ("H","N"):0.086,("H","P"):0.117,("H","Q"):0.083,("H","R"):0.062,("H","S"):0.128,
    ("H","T"):0.127,("H","V"):0.156,("H","W"):0.240,("H","Y"):0.150,("I","I"):0.000,
    ("I","K"):0.204,("I","L"):0.029,("I","M"):0.055,("I","N"):0.168,("I","P"):0.168,
    ("I","Q"):0.156,("I","R"):0.196,("I","S"):0.185,("I","T"):0.161,("I","V"):0.054,
    ("I","W"):0.215,("I","Y"):0.137,("K","K"):0.000,("K","L"):0.212,("K","M"):0.192,
    ("K","N"):0.097,("K","P"):0.159,("K","Q"):0.078,("K","R"):0.060,("K","S"):0.133,
    ("K","T"):0.124,("K","V"):0.184,("K","W"):0.319,("K","Y"):0.228,("L","L"):0.000,
    ("L","M"):0.050,("L","N"):0.176,("L","P"):0.170,("L","Q"):0.162,("L","R"):0.204,
    ("L","S"):0.191,("L","T"):0.168,("L","V"):0.062,("L","W"):0.210,("L","Y"):0.131,
    ("M","M"):0.000,("M","N"):0.159,("M","P"):0.165,("M","Q"):0.148,("M","R"):0.191,
    ("M","S"):0.180,("M","T"):0.157,("M","V"):0.095,("M","W"):0.233,("M","Y"):0.151,
    ("N","N"):0.000,("N","P"):0.149,("N","Q"):0.049,("N","R"):0.121,("N","S"):0.070,
    ("N","T"):0.088,("N","V"):0.148,("N","W"):0.296,("N","Y"):0.210,("P","P"):0.000,
    ("P","Q"):0.131,("P","R"):0.152,("P","S"):0.121,("P","T"):0.125,("P","V"):0.143,
    ("P","W"):0.288,("P","Y"):0.204,("Q","Q"):0.000,("Q","R"):0.087,("Q","S"):0.096,
    ("Q","T"):0.096,("Q","V"):0.165,("Q","W"):0.310,("Q","Y"):0.218,("R","R"):0.000,
    ("R","S"):0.141,("R","T"):0.136,("R","V"):0.184,("R","W"):0.266,("R","Y"):0.186,
    ("S","S"):0.000,("S","T"):0.112,("S","V"):0.156,("S","W"):0.320,("S","Y"):0.231,
    ("T","T"):0.000,("T","V"):0.113,("T","W"):0.269,("T","Y"):0.188,("V","V"):0.000,
    ("V","W"):0.278,("V","Y"):0.182,("W","W"):0.000,("W","Y"):0.130,("Y","Y"):0.000,
}

def _sw_dist(a: str, b: str) -> float:
    key = (min(a, b), max(a, b))
    return _SW_MATRIX.get(key, 0.0)


# ─── individual feature extractors ─────────────────────────────────────────

class AACExtractor:
    """Amino Acid Composition — 20 dim."""
    def extract(self, seq: str) -> np.ndarray:
        n = len(seq)
        vec = np.zeros(20, dtype=np.float64)
        for aa in seq:
            if aa in AA_INDEX:
                vec[AA_INDEX[aa]] += 1
        return vec / n if n > 0 else vec

    def feature_names(self) -> List[str]:
        return [f"AAC_{aa}" for aa in AAS]


class DPCExtractor:
    """Dipeptide Composition — 400 dim."""
    DIPEPTIDES = ["".join(p) for p in product(AAS, repeat=2)]
    DP_INDEX = {dp: i for i, dp in enumerate(DIPEPTIDES)}

    def extract(self, seq: str) -> np.ndarray:
        vec = np.zeros(400, dtype=np.float64)
        n = len(seq) - 1
        if n <= 0:
            return vec
        for i in range(n):
            dp = seq[i:i+2]
            if dp in self.DP_INDEX:
                vec[self.DP_INDEX[dp]] += 1
        return vec / n

    def feature_names(self) -> List[str]:
        return [f"DPC_{dp}" for dp in self.DIPEPTIDES]


class PAACExtractor:
    """Pseudo-Amino Acid Composition Type-1 — (20 + lambda) dim."""
    def __init__(self, lam: int = 10, weight: float = 0.05):
        self.lam = lam
        self.weight = weight

    def extract(self, seq: str) -> np.ndarray:
        L = len(seq)
        if L <= self.lam:
            # fallback: just AAC padded with zeros
            aac = AACExtractor().extract(seq)
            return np.concatenate([aac, np.zeros(self.lam)])

        # step 1: compute theta_k for k = 1..lambda
        thetas = []
        for k in range(1, self.lam + 1):
            total = 0.0
            for i in range(L - k):
                a, b = seq[i], seq[i + k]
                if a not in AA_INDEX or b not in AA_INDEX:
                    continue
                diff_sq = sum(
                    (prop[a] - prop[b]) ** 2 for prop in PAAC_PROPS
                ) / len(PAAC_PROPS)
                total += diff_sq
            thetas.append(total / (L - k))

        # step 2: AA frequencies
        aa_counts = np.zeros(20)
        for aa in seq:
            if aa in AA_INDEX:
                aa_counts[AA_INDEX[aa]] += 1
        f = aa_counts / L

        denom = f.sum() + self.weight * sum(thetas)
        p = f / denom

        tau = np.array([self.weight * t / denom for t in thetas])

        return np.concatenate([p, tau])

    def feature_names(self) -> List[str]:
        names = [f"PAAC_p{aa}" for aa in AAS]
        names += [f"PAAC_tau{k}" for k in range(1, self.lam + 1)]
        return names


class GAACExtractor:
    """Grouped Amino Acid Composition — 5 dim."""
    def extract(self, seq: str) -> np.ndarray:
        n = len(seq)
        vec = np.zeros(5, dtype=np.float64)
        for i, grp in enumerate(GAAC_ORDER):
            members = GAAC_GROUPS[grp]
            vec[i] = sum(1 for aa in seq if aa in members)
        return vec / n if n > 0 else vec

    def feature_names(self) -> List[str]:
        return [f"GAAC_{g}" for g in GAAC_ORDER]


class CTDCExtractor:
    """Composition — 21 dim (7 props × 3 classes)."""
    def extract(self, seq: str) -> np.ndarray:
        n = len(seq)
        vec = []
        for prop_name, groups in CTDC_PROPS.items():
            for cls in [1, 2, 3]:
                count = sum(1 for aa in seq if aa in groups[cls])
                vec.append(count / n if n > 0 else 0.0)
        return np.array(vec, dtype=np.float64)

    def feature_names(self) -> List[str]:
        names = []
        for prop_name in CTDC_PROPS:
            for cls in [1, 2, 3]:
                names.append(f"CTDC_{prop_name}_C{cls}")
        return names


class CTDTExtractor:
    """Transition — 21 dim (7 props × 3 transitions: 1↔2, 1↔3, 2↔3)."""
    TRANSITIONS = [(1, 2), (1, 3), (2, 3)]

    def extract(self, seq: str) -> np.ndarray:
        n = len(seq) - 1
        vec = []
        for prop_name, groups in CTDC_PROPS.items():
            # Map each residue to its class
            cls_seq = []
            for aa in seq:
                for cls, members in groups.items():
                    if aa in members:
                        cls_seq.append(cls)
                        break
                else:
                    cls_seq.append(0)
            for t1, t2 in self.TRANSITIONS:
                count = sum(
                    1 for i in range(n)
                    if (cls_seq[i] == t1 and cls_seq[i+1] == t2) or
                       (cls_seq[i] == t2 and cls_seq[i+1] == t1)
                )
                vec.append(count / n if n > 0 else 0.0)
        return np.array(vec, dtype=np.float64)

    def feature_names(self) -> List[str]:
        names = []
        for prop_name in CTDC_PROPS:
            for t1, t2 in self.TRANSITIONS:
                names.append(f"CTDT_{prop_name}_{t1}{t2}")
        return names


class CTDDExtractor:
    """
    Distribution — 105 dim (7 props × 3 classes × 5 percentiles).
    For each property and each class, record the fraction of the 
    sequence position where the 1st, 25th, 50th, 75th, 100th residue
    belonging to that class occurs.
    """
    PERCENTILES = [0, 25, 50, 75, 100]

    def extract(self, seq: str) -> np.ndarray:
        n = len(seq)
        vec = []
        for prop_name, groups in CTDC_PROPS.items():
            for cls in [1, 2, 3]:
                positions = [i for i, aa in enumerate(seq) if aa in groups[cls]]
                total = len(positions)
                if total == 0:
                    vec.extend([0.0] * 5)
                else:
                    for pct in self.PERCENTILES:
                        idx = max(0, int(np.ceil(pct / 100 * total)) - 1)
                        idx = min(idx, total - 1)
                        vec.append((positions[idx] + 1) / n)
        return np.array(vec, dtype=np.float64)

    def feature_names(self) -> List[str]:
        names = []
        for prop_name in CTDC_PROPS:
            for cls in [1, 2, 3]:
                for pct in self.PERCENTILES:
                    names.append(f"CTDD_{prop_name}_C{cls}_P{pct}")
        return names


class QSOExtractor:
    """
    Quasi-Sequence Order — (20 + 2*maxlag) dim.
    Uses Schneider-Wrede distance matrix (tau1) only for simplicity.
    """
    def __init__(self, maxlag: int = 30, weight: float = 0.1):
        self.maxlag = maxlag
        self.weight = weight

    def extract(self, seq: str) -> np.ndarray:
        L = len(seq)
        effective_lag = min(self.maxlag, L - 1)

        # sequence order coupling numbers
        tau = np.zeros(effective_lag, dtype=np.float64)
        for d in range(1, effective_lag + 1):
            total = 0.0
            for i in range(L - d):
                a, b = seq[i], seq[i + d]
                if a in AA_INDEX and b in AA_INDEX:
                    total += _sw_dist(a, b) ** 2
            tau[d - 1] = total / (L - d)

        # quasi-composition for each AA
        aa_counts = np.zeros(20, dtype=np.float64)
        for aa in seq:
            if aa in AA_INDEX:
                aa_counts[AA_INDEX[aa]] += 1
        f = aa_counts / L

        denom = 1.0 + self.weight * tau.sum()
        xr = f / denom
        xd = (self.weight * tau) / denom

        # Pad tau if sequence shorter than maxlag
        if effective_lag < self.maxlag:
            xd = np.concatenate([xd, np.zeros(self.maxlag - effective_lag)])

        return np.concatenate([xr, xd])

    def feature_names(self) -> List[str]:
        names = [f"QSO_xr_{aa}" for aa in AAS]
        names += [f"QSO_xd_{d}" for d in range(1, self.maxlag + 1)]
        return names


# ─── combined extractor ─────────────────────────────────────────────────────

class HandcraftedFeatureExtractor:
    """
    Combines all hand-crafted extractors into one pipeline.

    Total dimensions (default settings):
        AAC:  20
        DPC:  400
        PAAC: 30 (lambda=10)
        GAAC: 5
        CTDC: 21
        CTDT: 21
        CTDD: 105
        QSO:  50 (maxlag=20)
        ─────
        Total: 652
    """

    def __init__(self, paac_lambda: int = 10, paac_weight: float = 0.05,
                 qso_maxlag: int = 20, qso_weight: float = 0.1):
        self.extractors = [
            ("AAC",  AACExtractor()),
            ("DPC",  DPCExtractor()),
            ("PAAC", PAACExtractor(lam=paac_lambda, weight=paac_weight)),
            ("GAAC", GAACExtractor()),
            ("CTDC", CTDCExtractor()),
            ("CTDT", CTDTExtractor()),
            ("CTDD", CTDDExtractor()),
            ("QSO",  QSOExtractor(maxlag=qso_maxlag, weight=qso_weight)),
        ]

    def extract(self, seq: str) -> np.ndarray:
        parts = [ext.extract(seq) for _, ext in self.extractors]
        return np.concatenate(parts)

    def extract_batch(self, sequences: List[str],
                      show_progress: bool = True) -> np.ndarray:
        from tqdm import tqdm
        it = tqdm(sequences, desc="Hand-crafted features") if show_progress else sequences
        return np.stack([self.extract(seq) for seq in it])

    def feature_names(self) -> List[str]:
        names = []
        for _, ext in self.extractors:
            names.extend(ext.feature_names())
        return names

    @property
    def n_features(self) -> int:
        return len(self.feature_names())
