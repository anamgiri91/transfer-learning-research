"""The in-domain 3C / 3C-like protease corpus: what is in it, and what is not.

ChEMBL indexes picornaviral proteases under the **whole genome polyprotein**,
whose component synonyms cover 2A, 3C, capsid and RNA polymerase alike. A
target id is therefore not evidence of which enzyme was assayed, and the screen
below -- on the assay description -- is what actually decides corpus membership.
It is domain logic, not script glue, which is why it lives here and is tested.
"""
from __future__ import annotations

import re

import pandas as pd

# label -> (chembl target id, virus family). Verified against the target
# endpoint on 2026-09-02; manifest counts let a rebuild detect drift.
TARGETS = {
    "sars2_3cl":  ("CHEMBL4523582", "coronaviral"),
    "sars1_3cl":  ("CHEMBL3927",    "coronaviral"),
    "mers_3cl":   ("CHEMBL4295557", "coronaviral"),
    "ibv_3cl":    ("CHEMBL1293307", "coronaviral"),
    "cvb3_3c":    ("CHEMBL2396505", "picornaviral"),
    "ev71_3c":    ("CHEMBL4295525", "picornaviral"),
    "hrv14_3c":   ("CHEMBL4295564", "picornaviral"),
    "hrv16_3c":   ("CHEMBL5296",    "picornaviral"),
}

# Excluded with the reason stated, rather than quietly absent from TARGETS.
EXCLUDED = {
    "CHEMBL5127": "poliovirus entry is RNA-polymerase data, not a protease",
    "CHEMBL3232683": "HCoV-NL63 entry is PLP2, a papain-like protease -- different fold",
}

# An assay counts as 3C / 3C-like only if it says so. Cell-based antiviral and
# capsid readouts are a different measurement and plan.md §3.2 forbids pooling
# them. DROP wins over KEEP: a description naming both folds is ambiguous, and
# being conservative costs a few records but keeps the wrong fold out.
KEEP = re.compile(r"3c[\s-]*like|3cl|\b3c\b|main protease|mpro|picornain", re.I)
DROP = re.compile(r"papain|plpro|plp2|\bpl-?pro\b|polymerase|capsid|2a\s*protease|"
                  r"helicase|methyltransferase|cytotox|cell viability", re.I)


def screen(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep only records whose assay description names a 3C / 3C-like protease."""
    if df.empty:
        return df, {"kept": 0, "dropped_not_3c": 0, "dropped_wrong_enzyme": 0}
    desc = df.get("assay_description", pd.Series([""] * len(df))).fillna("")
    wrong = desc.str.contains(DROP)
    named = desc.str.contains(KEEP)
    keep = named & ~wrong
    return df[keep].copy(), {
        "kept": int(keep.sum()),
        "dropped_not_3c": int((~named & ~wrong).sum()),
        "dropped_wrong_enzyme": int(wrong.sum()),
    }
