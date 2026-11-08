# Time-Series-Library (vendor edilmiş alt küme)

Kaynak: <https://github.com/thuml/Time-Series-Library> (THUML, MIT; `LICENSE`)
Commit: `4e938a1767106324dd753b2a44832bf870a0252e` (2026-04-18) · Kopyalandı: 2026-10-19

TSLib bir `pip` paketi değil ve kullanılmayan bağımlılıklar istiyor (reformer, sktime, patool). Bu yüzden yalnızca
M2 modellerinin gerektirdiği dosyalar kopyalandı. Modeller `models/tslib_adapter.py` üzerinden kullanılır; bu
klasördeki dosyalara doğrudan dokunulmaz.

| Klasör | Dosyalar |
|---|---|
| `models/` | `PatchTST.py`, `iTransformer.py`, `TimesNet.py`, `TimeMixer.py`, `TSMixer.py` |
| `layers/` | `Transformer_EncDec.py`, `SelfAttention_Family.py`, `Embed.py`, `Conv_Blocks.py`, `Autoformer_EncDec.py`, `StandardNorm.py` |
| `utils/` | `masking.py` |

Vanilla Transformer TSLib'den alınmadı; kendi kodumuz (`models/sequence.py`).

## Değişiklikler

Dosyalar kaynakla birebir aynı; yalnızca import satırları değişti:

- `from layers.X import ...` → `from ..layers.X import ...`, `from utils.masking import ...` →
  `from ..utils.masking import ...` (paket içi göreli import).
- `layers/SelfAttention_Family.py`: `from reformer_pytorch import LSHSelfAttention` satırı silindi. Paket kurulu
  değil; yalnızca kullanılmayan `ReformerLayer` sınıfı ona dayanıyor.
- Boş `__init__.py` dosyaları eklendi.

Doğrulama: aynı commit'i klonlayıp `diff -r` ile karşılaştırınca yalnızca bu satırlar farklı çıkmalı.
