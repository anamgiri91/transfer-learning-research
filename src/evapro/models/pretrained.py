"""Immutable checkpoint identity used by every executed ChemBERTa arm.

Pinned to the snapshot present in the original study's Hugging Face cache.
"""
CHEMBERTA = "DeepChem/ChemBERTa-77M-MTR"
CHEMBERTA_REVISION = "66b895cab8adebea0cb59a8effa66b2020f204ca"


def checkpoint_kwargs(model_name: str = CHEMBERTA) -> dict[str, str]:
    if model_name != CHEMBERTA:
        raise ValueError("The reproducible study supports only the pinned ChemBERTa-77M-MTR checkpoint")
    return {"revision": CHEMBERTA_REVISION}
