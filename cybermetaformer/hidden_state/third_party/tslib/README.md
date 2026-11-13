# Time-Series-Library (vendored subset)

Source: <https://github.com/thuml/Time-Series-Library> (THUML, MIT; `LICENSE`)
Commit: `4e938a1767106324dd753b2a44832bf870a0252e` (2026-04-18) · Copied: 2026-10-19

TSLib is not a `pip` package and requires dependencies we do not use (reformer, sktime, patool). Only the files the
M2 models need were therefore copied. The models are used through `models/tslib_adapter.py`; the files in this folder
are not edited directly.

| Folder | Files |
|---|---|
| `models/` | `PatchTST.py`, `iTransformer.py`, `TimesNet.py`, `TimeMixer.py`, `TSMixer.py` |
| `layers/` | `Transformer_EncDec.py`, `SelfAttention_Family.py`, `Embed.py`, `Conv_Blocks.py`, `Autoformer_EncDec.py`, `StandardNorm.py` |
| `utils/` | `masking.py` |

The vanilla Transformer is not taken from TSLib; it is our own code (`models/sequence.py`).

## Changes

The files are identical to the source; only the import lines changed:

- `from layers.X import ...` → `from ..layers.X import ...`, `from utils.masking import ...` →
  `from ..utils.masking import ...` (relative imports within the package).
- `layers/SelfAttention_Family.py`: the line `from reformer_pytorch import LSHSelfAttention` was removed. The package
  is not installed; only the unused `ReformerLayer` class depends on it.
- Empty `__init__.py` files were added.

Check: cloning the same commit and comparing with `diff -r` should show only these lines as different.
