# MiniCAGE Otonom Siber Savunma Veri Seti (minicage_v1)

> **Bir paragrafta:** Bu veri seti, bir saldırganın (Red) küçük bir kurumsal ağda ilerlediği, bir savunmacının (Blue) da
> bunu durdurmaya çalıştığı **24 000 simüle edilmiş saldırı bölümünden** oluşur (toplam **2 400 000 zaman adımı**).
> Her adımda savunmacının gördüğü **kısmi ve gürültülü gözlem** ile simülatörün bildiği **gizli gerçek durum**
> (hangi makine ele geçirildi, saldırganın yetki düzeyi ne) birlikte kaydedilir. Böylece gerçek dünyada çok pahalı olan
> kusursuz etiketler kendiliğinden elde edilir. Aynı ham kayıttan denetimli öğrenme (saldırı tespiti, erken uyarı,
> saldırgan modelleme), dizi modelleme, offline pekiştirmeli öğrenme ve dağılım kayması çalışmaları için görevler türetilir.

| Bir bakışta | |
|---|---|
| Kaynak | CybORG++ / MiniCAGE simülatörü (CAGE Challenge 2 senaryosu) |
| Büyüklük | 24 000 bölüm × 100 adım = 2 400 000 geçiş; 33 MB (sıkıştırılmış) |
| Öğrenme türleri | çok etiketli sınıflandırma, ikili sınıflandırma, dizi sınıflandırma, offline RL / taklit, dinamik modelleme |
| Etiket kalitesi | simülatörden alınan kesin gerçek durum (insan etiketlemesi yok) |
| Bölümler | train 16 000, val 2 000, test 2 000, OOD test 4 000 bölüm |
| Üretim | tek CPU çekirdeğinde ≈23 s; tamamen yeniden üretilebilir (sabit seed) |
| Format | NumPy `.npz` (ham), `csv.gz` (tablo), Python yükleyicileri (`experiments/dataset/loaders.py`) |

---

## 1. Motivasyon

**Otonom siber savunma**, bir ağı saldırılara karşı insan müdahalesi olmadan (ya da azaltılmış müdahaleyle) savunan
öğrenen ajanlar geliştirmeyi hedefler. Bu alanda ortak bir test zemini olarak **CAGE Challenge 2** ve onun simülatörü
**CybORG** yaygın biçimde kullanılır. **CybORG++** (Emerson vd., 2024) bu simülatörün hataları ayıklanmış bir sürümünü
ve ~1000 kat daha hızlı, vektörleştirilmiş bir yeniden uygulaması olan **MiniCAGE**'i sunar.

Gerçek saldırı tespit (IDS) veri setleriyle çalışırken üç temel sorun vardır:

1. **Etiketler pahalı ve gürültülüdür.** Bir makinenin gerçekten ele geçirilip geçirilmediği çoğu zaman kesin bilinmez.
2. **Saldırganın ilerleyişi görünmezdir.** Kayıtlar tekil olayları içerir; saldırının hangi aşamada olduğu bilgisi yoktur.
3. **Veri statiktir.** Savunmacının aksiyonları gelecekteki veriyi etkilemez, dolayısıyla karar verme (RL) çalışılamaz.

Simülasyonla üretilen bu veri seti bu üç sorunu aşar: simülatör gerçek durumu bilir (kesin etiket), saldırının her adımı
kayıtlıdır, ve savunmacının aksiyonları gelecekteki durumları değiştirir (etkileşimli veri). Bedeli ise
**gerçekçilik**tir: ağ küçük ve soyut, saldırganlar betik tabanlıdır (bkz. §10). Alandaki diğer veri setleri ve
çalışmalarla karşılaştırma için bkz. §9.

**Biçimsel çerçeve.** Problem, kısmen gözlemlenebilir bir Markov karar süreci (POMDP) olarak düşünülebilir:
her adım t'de gizli durum `s_t` (13 makinenin gerçek ele geçirilme durumu), savunmacının gözlemi `o_t` (78 boyut),
savunmacı aksiyonu `a_t` (53 seçenek), saldırgan aksiyonu ve ödül `r_t` vardır. Bu veri setinin ayırt edici
özelliği, normalde gizli olan `s_t`'nin de kaydedilmesidir.

---

## 2. Simülasyon ortamı

### 2.1 Ağ

13 makine (host) üç alt ağa ayrılır:

| Alt ağ | Hostlar | Rol |
|---|---|---|
| Kullanıcı | `user0` … `user4` | Saldırının başladığı yer. `user0` saldırganın kalıcı dayanağıdır, savunmacı ona müdahale edemez. |
| Kurumsal | `ent0`, `ent1`, `ent2`, `def` | Ara katman. `def` savunmacının kendi makinesidir ve hiç ele geçirilmez. |
| Operasyonel | `ophost0` … `ophost2`, `opserv` | **`opserv` (operasyonel sunucu) kritik varlıktır.** Saldırganın nihai hedefi. |

Saldırı yolu (MiniCAGE bağlantıları):

```
user0 (Red dayanağı) ─► user1/user2 ─► ent0 ─┐
                     └► user3/user4 ─► ent1 ─┴─► ent2 ─► opserv ─► IMPACT (hizmet kesintisi)
```

### 2.2 Ajanlar

| Ajan | Rol | Bu veri setinde |
|---|---|---|
| **Red** (saldırgan) | Keşif → istismar (exploit) → yetki yükseltme (escalate) zinciriyle ilerler, `opserv`'e ulaşınca impact uygular. | İki betik tabanlı ajan: **B-line** (hedefe en kısa yoldan gider) ve **Meander** (her alt ağı keşfederek yavaş yayılır). |
| **Blue** (savunmacı) | Hostları analiz eder, tuzak (decoy) servis kurar, saldırganı temizler ya da hostu geri yükler (restore). | Altı davranış politikası (§3.1). |
| **Green** (normal kullanıcı) | Her adımda %50 olasılıkla rastgele bir hostu tarar. | Gözlemde **yanlış alarm** gürültüsü üretir; ayrıca kaydedilmez. |

Red ve Blue **eş zamanlı** hamle yapar: ikisi de aynı durumu görerek karar verir. Bu nedenle savunmacı her zaman bir
adım geriden gelir. Çakışan hamlelerde öncelik kuralları vardır (ör. aynı adımda exploit ve restore → exploit kazanır).

### 2.3 Aksiyonlar

**Blue (53 aksiyon):** `host = (a − 1) mod 13`

| İndeks | Aksiyon | Etki |
|---|---|---|
| 0 | `sleep` | Hiçbir şey yapma |
| 1–13 | `analyse_<host>` | Hostta saldırganın **root (yönetici) yetkisi** olup olmadığını gözleme yansıtır |
| 14–26 | `decoy_<host>` | Hosta tuzak servis kur; saldırgan o servisi istismar etmeye çalışırsa başarısız olur |
| 27–39 | `remove_<host>` | Kullanıcı düzeyindeki saldırgan erişimini kaldırır (root'a etkisizdir) |
| 40–52 | `restore_<host>` | Hostu temiz duruma geri yükler (her kullanımda −1 ödül maliyeti) |

**Red (56 aksiyon):** `sleep`, 3 alt ağ keşfi (`remote_sub*`) ve her host için `network` (servis tarama), `exploit`,
`escalate`, `impact` (13 × 4). Adlar `meta.json → action_names` içindedir.

### 2.4 Savunmacının gözlemi (78 boyut)

| Boyutlar | İçerik | Değerler |
|---|---|---|
| 0–51 | Host başına 4 bit: **aktivite** [tarama, istismar] + **güvenlik** durumu | güvenlik: `[0,0]` temiz/bilinmiyor, `[0,1]` istismar edildi, `[1,1]` root (yalnızca `analyse` ile görülür), `[1,0]` remove uygulandı |
| 52–64 | Host başına tarama geçmişi | 0 hiç, 1 daha önce, 2 bu adımda |
| 65–77 | Host başına kalan tuzak (decoy) sayısı | 0–4 |

Gözlemi gerçek durumdan ayıran **gürültü kaynakları**: istismarlar yalnızca %95 olasılıkla görünür; Green taramaları
sahte tarama aktivitesi üretir; root yetkisi `analyse` yapılmadan görünmez; ve (§5.3) uyarı bayrakları saldırı
sürerken kaybolabilir.

### 2.5 Ödül (Blue perspektifi, her adım)

| Durum | Ödül |
|---|---|
| Ele geçirilmiş her kullanıcı / operasyonel host (`opserv` hariç; kullanıcı hostlarında yalnızca bazı istismar türleri) | −0.1 |
| Ele geçirilmiş her kurumsal host ve `opserv` | −1 |
| `opserv` impact altında | −10 |
| Her `restore` kullanımı | −1 |

Ceza ele geçirme sürdükçe **her adım** tekrarlanır. Red'in ödülü Blue'nunkinin negatifidir (sıfır toplamlı oyun).

### 2.6 Örnek bir bölüm

B-line saldırganı × React-Restore savunmacısı, ilk 15 adım (bölüm #2000). "Gerçek durum" modelin göremediği etikettir; "Uyarılar" savunmacının gözleminde istismar bayrağı açık olan hostlardır. ✓/✗ saldırganın aksiyonunun başarılı olup olmadığını gösterir.

| Adım | Red aksiyonu | Blue aksiyonu | Ödül | Gerçek durum (gizli) | Uyarılar (gözlem) |
|---|---|---|---|---|---|
| 0 | `remote_sub2` ✓ | `sleep` | 0.0 | — | — |
| 1 | `network_user1` ✓ | `sleep` | 0.0 | — | — |
| 2 | `exploit_user1` ✓ | `sleep` | 0.0 | user1(user) | user1 |
| 3 | `escalate_user1` ✓ | `restore_user1` | -1.0 | — | — |
| 4 | `exploit_user1` ✓ | `sleep` | -0.1 | user1(user) | — |
| 5 | `escalate_user1` ✓ | `sleep` | -0.1 | user1(root) | — |
| 6 | `network_ent0` ✓ | `sleep` | -0.1 | user1(root) | — |
| 7 | `exploit_ent0` ✓ | `sleep` | -0.1 | ent0(user), user1(root) | ent0 |
| 8 | `escalate_ent0` ✓ | `restore_ent0` | -1.1 | user1(root) | — |
| 9 | `exploit_ent0` ✓ | `sleep` | -0.1 | ent0(user), user1(root) | ent0 |
| 10 | `escalate_ent0` ✓ | `restore_ent0` | -1.1 | user1(root) | — |
| 11 | `exploit_ent0` ✓ | `sleep` | -0.1 | ent0(user), user1(root) | ent0 |
| 12 | `escalate_ent0` ✓ | `restore_ent0` | -1.1 | user1(root) | — |
| 13 | `exploit_ent0` ✓ | `sleep` | -0.1 | ent0(user), user1(root) | — |
| 14 | `escalate_ent0` ✓ | `sleep` | -1.1 | ent0(root), user1(root) | — |

Bu tabloda veri setinin iki temel özelliği görülebilir: (1) Savunmacı uyarı gördüğü hostu restore eder, saldırgan aynı
hostu tekrar istismar eder. Bu "kedi-fare" döngüsü bölüm boyunca sürebilir. (2) Bazı adımlarda gerçek durumda ihlal
vardır ama gözlemde uyarı yoktur (ör. restore sonrası yeniden ele geçirilen hostlar). Tespit modellerinin çözmesi
gereken asıl zorluk budur (§5.3).

---

## 3. Veri üretimi

### 3.1 Davranış politikaları

Veri çeşitliliği için her saldırgan, farklı kalitede altı savunmacı politikasıyla eşleştirildi. Bu yaklaşım
offline RL kıyaslama setlerindeki (ör. D4RL) "rastgele / orta / uzman" veri karışımına benzer.

| Blue politikası | Davranış | Neden var |
|---|---|---|
| `sleep` | Hiçbir şey yapmaz | Savunmasız referans; saldırı zincirinin tamamı gözlenir |
| `react_restore` | İstismar uyarısı gördüğü hostu restore eder | Reaktif, basit savunma |
| `restore_decoys` | `react_restore` + boşta kaldığında kritik hostlara tuzak kurar | Bu setteki en iyi savunma ("uzman" rolü) |
| `random` | 53 aksiyondan rastgele seçer | Aksiyon uzayını geniş kapsar (offline RL için önemli), düşük kalite |
| `react_restore_eps0.1` | %10 olasılıkla rastgele aksiyon | Gürültülü uzman |
| `react_restore_eps0.3` | %30 olasılıkla rastgele aksiyon | Daha gürültülü uzman; **dağılım dışı (OOD) test için ayrılmıştır** |

Izgara: 2 Red × 6 Blue × 2000 bölüm, her bölüm 100 adım, `remove_bugs=True`
(MiniCAGE'in orijinal CAGE 2 hatalarını taklit etmeyen modu). Her (Red, Blue) kombinasyonu 1000 bölümlük
"shard"lar hâlinde, her shard kendi seed'iyle üretildi.

### 3.2 Bölümler (train / val / test)

Bölme **bölüm (episode) düzeyinde** yapılır: aynı bölümün adımları farklı kümelere düşmez (zamansal sızıntı yok).
Train/val/test her (Red, Blue) kombinasyonuna göre tabakalıdır (%80/%10/%10). `react_restore_eps0.3` politikasının
tüm bölümleri yalnızca `ood_test` kümesindedir; böylece "eğitimde hiç görülmemiş bir savunmacı davranışına genelleme"
ölçülebilir.

| Küme | Bölüm | Adım |
|---|---|---|
| train | 16 000 | 1 600 000 |
| val | 2 000 | 200 000 |
| test | 2 000 | 200 000 |
| OOD test | 4 000 | 400 000 |

---

## 4. Veri yapısı

### 4.1 Ham alanlar (bölüm-ana: E bölüm, T adım)

| Alan | Şekil | Açıklama |
|---|---|---|
| `obs_blue` | (E, T+1, 78) | Blue gözlemi (§2.4) |
| `obs_red` | (E, T+1, 40) | Red gözlemi: başarı bayrağı + 13 host × [taranmış, user, root] |
| `true_state` | (E, T+1, 39) | **Gizli gerçek durum**: 13 host × [taranmış, user erişimi, root erişimi] (−1 = Red henüz bilmiyor) |
| `act_blue` / `act_red` | (E, T) | Seçilen aksiyonlar (53 / 56) |
| `rew_blue` | (E, T) | Blue ödülü |
| `blue_success` / `red_success` | (E, T) | Aksiyon başarılı mı (−1 = uygulanamaz) |
| `impacted` | (E, T, 13) | Impact durumu (yalnızca `opserv` ödülü etkiler) |
| `decoys_left` | (E, T+1, 13) | Host başına kalan tuzak |
| `red_policy`, `blue_policy`, `epsilon`, `remove_bugs`, `seed` | (E,) | Bölüm meta verisi |

**Zaman hizalaması:** `obs_*[:, t]` ve `true_state[:, t]` t. adımdan *sonraki* durumdur (t = 0 başlangıç).
`act_*[:, t]` bu durum görülerek seçilen aksiyondur, `rew_blue[:, t]` onun sonucudur.

### 4.2 Türetilmiş etiketler

- **Host durumu:** `true_state`'ten host başına 3 sınıf: **0 temiz**, **1 kullanıcı erişimi**, **2 root erişimi**.
  `user0` (saldırganın kalıcı dayanağı) etiketlerden çıkarılır → 12 host.
- **Ağda ihlal var mı:** 12 hosttan en az biri temiz değilse 1.
- **Red aksiyon tipi:** 56 aksiyon → 6 tip (sleep, remote, network, exploit, escalate, impact).
- **Erken uyarı:** `opserv` önümüzdeki h adımda impact edilecek mi.

### 4.3 Dosyalar

| Dosya | İçerik |
|---|---|
| `shard_###.npz` | Ham kayıt, bir (Red, Blue) kombinasyonu × 1000 bölüm |
| `meta.json` | Aksiyon adları, host listesi, gözlem düzeni, shard listesi, sürümler |
| `splits.npz` | Bölüm indeksleri: `train`, `val`, `test`, `ood_test` |
| `tabular.csv.gz` | Adım başına okunabilir satır (2 400 000 satır): aksiyon adları, ödül, ele geçirilmiş host sayıları, host başına uyarı ve gerçek etiket |
| `baselines.json` | Referans model sonuçları (§8) |

### 4.4 Kullanım

```python
from experiments.dataset import loaders as L

raw, meta = L.load_raw('experiments/data/minicage_v1')   # tüm alanlar, ~0.5 GB RAM
sp = L.load_splits('experiments/data/minicage_v1')

X, y, y_any = L.detection_view(raw, sp['train'], k=4)     # tespit: (N, 312), (N, 12), (N,)
X, y = L.risk_view(raw, sp['train'], horizon=10)          # erken uyarı
X, y = L.attacker_id_view(raw, sp['train'], n_steps=10)   # saldırgan tanıma: (E, 10, 78)
D = L.d4rl_view(raw, sp['train'])                         # offline RL (D4RL sözlüğü)
D = L.dynamics_view(raw, sp['train'])                     # dünya modeli
```

Tablo hâli için (pandas ile): `pd.read_csv('tabular.csv.gz')`.

---

## 5. Veri setinin istatistikleri

### 5.1 Saldırgan × savunmacı kombinasyonları

| Red | Blue | Bölüm | Bölüm ödülü (ort. ± SE) | Impact % | İlk impact (adım) | Son ihlal (host) |
|---|---|---|---|---|---|---|
| bline | sleep | 2 000 | -1134.6 ± 0.4 | 100.0 | 14 | 4.00 |
| bline | react_restore | 2 000 | -125.4 ± 1.2 | 0.0 | — | 2.68 |
| bline | restore_decoys | 2 000 | -54.6 ± 0.4 | 0.0 | — | 0.93 |
| bline | random | 2 000 | -828.1 ± 7.4 | 82.2 | 14 | 1.51 |
| bline | react_restore_eps0.1 | 2 000 | -124.1 ± 2.0 | 26.3 | 72 | 1.69 |
| bline | react_restore_eps0.3 | 2 000 | -178.7 ± 3.5 | 84.0 | 41 | 0.98 |
| meander | sleep | 2 000 | -1061.4 ± 0.6 | 100.0 | 28 | 7.32 |
| meander | react_restore | 2 000 | -117.1 ± 1.1 | 0.0 | — | 4.28 |
| meander | restore_decoys | 2 000 | -85.1 ± 0.8 | 0.0 | — | 4.15 |
| meander | random | 2 000 | -749.3 ± 2.7 | 99.9 | 33 | 4.35 |
| meander | react_restore_eps0.1 | 2 000 | -106.5 ± 1.2 | 5.1 | 71 | 3.38 |
| meander | react_restore_eps0.3 | 2 000 | -107.9 ± 1.6 | 31.8 | 66 | 2.48 |

*Impact %: `opserv`'in en az bir kez impact edildiği bölüm oranı. İlk impact: medyan adım. Son ihlal: bölüm sonunda
ele geçirilmiş host sayısı (ortalama, `user0` hariç).*

### 5.2 Etiket dağılımı (adım × host oranı)

| Host | temiz | kullanıcı | root |
|---|---|---|---|
| `def` | 1.000 | 0.000 | 0.000 |
| `ent0` | 0.729 | 0.039 | 0.232 |
| `ent1` | 0.705 | 0.031 | 0.264 |
| `ent2` | 0.679 | 0.028 | 0.293 |
| `ophost0` | 0.969 | 0.004 | 0.027 |
| `ophost1` | 0.969 | 0.004 | 0.027 |
| `ophost2` | 0.968 | 0.004 | 0.028 |
| `opserv` | 0.677 | 0.067 | 0.256 |
| `user1` | 0.708 | 0.031 | 0.261 |
| `user2` | 0.711 | 0.030 | 0.259 |
| `user3` | 0.708 | 0.032 | 0.260 |
| `user4` | 0.708 | 0.032 | 0.259 |

Ağda en az bir ihlal olan adım oranı: **0.851**. Sınıflar dengesizdir: `ophost*` neredeyse her zaman temizdir,
"kullanıcı erişimi" sınıfı kısa süreli bir ara durumdur. Makro-F1 veya sınıf ağırlıklandırma önerilir.
`def` hiç ele geçirilmez (sabit 0).

### 5.3 Uyarılar gerçek durumu ne kadar yansıtıyor?

Gözlemdeki "istismar edildi" bayrağını basit bir uyarı olarak ele alırsak:

- Bölümler boyunca **138 334** kez bir host ele geçirildi. Bunların **%94.2**'i en az bir kez uyarı üretti. Uyarı genellikle aynı adımda geliyor (medyan gecikme 0 adım, ortalama 0.59 adım).
- İhlalin **başladığı** adımda uyarı oranı: ilk ele geçirmede %91.7, restore sonrası yeniden ele geçirmede %93.0.
- **Adım bazında** ise tablo farklı: uyarı varsa ihlal kesin (kesinlik 1.000), ama ihlal süren adımların yalnızca **0.576**'ünde uyarı açık (duyarlılık).
- `react_restore` bölümlerinde uyarısız ihlal adımlarının **%79**'inde o host daha önce restore edilmiş.

Adım bazında duyarlılığın savunmacı politikasına göre değişimi:

| Blue politikası | Uyarı duyarlılığı |
|---|---|
| `sleep` | 0.913 |
| `react_restore` | 0.194 |
| `restore_decoys` | 0.248 |
| `random` | 0.709 |
| `react_restore_eps0.1` | 0.264 |
| `react_restore_eps0.3` | 0.406 |

**Yorum:** Uyarılar ihlalin *başlangıcını* iyi yakalıyor, ancak savunmacı aktif olduğunda (özellikle restore sonrası
yeniden ele geçirilen hostlarda) bayrak ihlal sürerken kayboluyor. Yani **anlık gözlem gerçek durumu belirlemeye yetmez;
geçmişi hatırlayan modeller gerekir.** Ayrıca bu ilişki savunmacı politikasına bağlıdır. Bu da tespit görevini politika
değişimine karşı hassas yapar (§6.6). Bu olgu veriden ölçülmüştür; simülatör içindeki nedeni ayrıntılı incelenmemiştir.

---

## 6. Bu veri setiyle neler yapılabilir?

Aşağıdaki görevlerin tamamı aynı ham kayıttan türetilir. "Hazır" sütunu yükleyicinin ve referans sonucun var olup
olmadığını gösterir.

| # | Görev | ML türü | Zorluk | Hazır |
|---|---|---|---|---|
| 6.1 | Saldırı tespiti / gizli durum kestirimi | Çok etiketli sınıflandırma, dizi modelleme | Orta | yükleyici + sonuç |
| 6.2 | OpServer için erken uyarı | Dengesiz ikili sınıflandırma | Orta–zor (OOD'da zor) | yükleyici + sonuç |
| 6.3 | Saldırgan modelleme | Dizi sınıflandırma, sıradaki aksiyon tahmini | Şu an kolay | yükleyici + sonuç |
| 6.4 | Offline RL ve taklit öğrenme | Offline RL, davranış klonlama | Orta–zor | yükleyici + BC sonucu |
| 6.5 | Dünya modeli / dinamik öğrenme | Çok çıktılı regresyon / sınıflandırma | Orta | yükleyici |
| 6.6 | Dağılım kayması ve genelleme | Alan genellemesi, OOD tespiti | Zor | bölüm + sonuçlar |
| 6.7 | Tablo verisi ve yorumlanabilirlik | Gradient boosting, SHAP, EDA | Kolay–orta | CSV |
| 6.8 | Etiket verimliliği / yarı denetimli öğrenme | Az etiketle öğrenme, aktif öğrenme | Orta | ham veri |

### 6.1 Saldırı tespiti ve gizli durum kestirimi

- **Problem:** Savunmacının son k gözleminden (`k × 78` özellik) her hostun gerçek durumunu {temiz, kullanıcı, root}
  tahmin et. İkili sürüm: "ağda herhangi bir ihlal var mı?"
- **Neden ilginç:** Klasik IDS problemine benzer, ama etiketler kesin. Gözlem gürültülü ve eksik (§5.3), root yetkisi
  doğrudan görünmez. Bu açıdan bir POMDP'de **inanç durumu (belief state) kestirimi** problemidir.
- **Önerilen yöntemler:** MLP (pencereli girdi), GRU/LSTM/Transformer (tüm geçmiş), host'lar arası ilişki için grafik
  sinir ağı (ağ topolojisi bilindiği için doğal bir aday), olasılıksal filtreleme (HMM / parçacık filtresi) ile kıyas. CAGE-2 için parçacık filtresi tabanlı bir inanç
  durumu yaklaşımı Le & Stadler (2025) tarafından önerilmiştir (§9.3).
- **Metrikler:** Host başına makro-F1, ağ düzeyinde F1, tespit gecikmesi (ihlalden ilk doğru tahmine kadar geçen adım).
- **Referans sonuç** (k = modelin gördüğü son gözlem sayısı):

| Model | Host makro-F1 (test) | Host makro-F1 (OOD) | İhlal F1 (test) | İhlal F1 (OOD) |
|---|---|---|---|---|
| Çoğunluk sınıfı | 0.285 | 0.311 | — | — |
| Uyarı kuralı | — | — | 0.833 | 0.802 |
| MLP, k=1 | 0.910 | 0.808 | 0.940 | 0.889 |
| MLP, k=4 | 0.920 | 0.810 | 0.961 | 0.900 |
| MLP, k=8 | 0.929 | 0.816 | 0.969 | 0.899 |

  Basit uyarı kuralı (F1 0.833) ile kıyaslandığında öğrenen model belirgin şekilde daha iyi. Geçmiş penceresini
  büyütmek (k = 1 → 8) test kümesinde ağ düzeyinde F1'i 0.940 → 0.969 yükseltiyor; görülmemiş politikada ise performans
  0.899 seviyesine düşüyor.
- **Açık sorular:** Uzun geçmişi kullanan dizi modelleri kayıp bayrak sorununu çözebilir mi? Kalibre edilmiş
  olasılıklar elde edilebilir mi (savunmacının karar vermesi için önemli)?

### 6.2 Erken uyarı: kritik sunucu düşmeden önce haber vermek

- **Problem:** Şu ana kadarki gözlemlerden, `opserv` önümüzdeki 10 adım içinde impact edilecek mi?
  (Zaten impact altındaki adımlar çıkarılır; pozitif oranı ≈ 0.078.)
- **Neden ilginç:** Dengesiz, ileriye dönük (prognostik) bir sınıflandırma problemidir. Pratikte en değerli soru
  "şu an ne oldu" değil "birazdan ne olacak"tır.
- **Metrikler:** AUC, F1, belirli bir yanlış alarm oranında duyarlılık, uyarının ne kadar önceden verildiği.
- **Referans sonuç:**

| Model | F1 (test) | AUC (test) | F1 (OOD) | AUC (OOD) |
|---|---|---|---|---|
| Uyarı kuralı (`ent2`/`opserv` uyarısı) | 0.360 | — | 0.385 | — |
| MLP, k=4 | 0.796 | 0.983 | 0.198 | 0.753 |

  Eğitimle aynı dağılımda model çok iyi (AUC 0.983), ancak **eğitimde görülmemiş savunmacı politikasında
  AUC 0.753'e, F1 0.796 → 0.198 düşüyor** ve basit kuralın bile gerisinde kalıyor.
  Bu, verinin dağılım kayması çalışmaları için iyi bir test zemini olduğunu gösteriyor (§6.6).

### 6.3 Saldırgan modelleme

- **Problemler:** (a) İlk n adımın gözlem dizisinden saldırganın türünü (B-line / Meander) tanı.
  (b) Gözlem geçmişinden saldırganın sıradaki aksiyon tipini tahmin et.
- **Referans sonuç:**

(a) Saldırgan tanıma, doğruluk:

| Model | test | OOD test |
|---|---|---|
| Şans | 0.500 | 0.500 |
| GRU, ilk 3 adım | 0.998 | 0.995 |
| GRU, ilk 5 adım | 1.000 | 1.000 |
| GRU, ilk 10 adım | 1.000 | 0.999 |

(b) Red aksiyon tipi (6 sınıf) tahmini, doğruluk:

| Model | test | OOD test |
|---|---|---|
| Çoğunluk sınıfı | 0.326 | 0.417 |
| MLP, k=4 | 0.969 | 0.949 |

- **Değerlendirme:** Her iki görev de **şu haliyle kolay**, çünkü iki saldırgan da deterministik betiklerdir ve
  ilk birkaç adımda farklı davranırlar. Görevi anlamlı kılmak için üretime gürültülü (ε) saldırganlar, karışık
  stratejiler veya öğrenen (PPO) saldırganlar eklenmelidir (§11). Bu hâliyle dizi modelleri için bir "akıl sağlığı
  testi" olarak kullanılabilir. Red aksiyon tahmininin CAGE-2'deki doğrudan karşılığı için bkz. Samaddar vd. (2026),
  §9.3.

### 6.4 Offline pekiştirmeli öğrenme ve taklit öğrenme

- **Problem:** Ortamla etkileşime girmeden, yalnızca kaydedilmiş geçişlerden (`gözlem, aksiyon, ödül, sonraki gözlem`)
  iyi bir savunma politikası öğren.
- **Neden ilginç:** Gerçek ağlarda deneme-yanılma ile savunma öğrenmek kabul edilemez; offline RL bu yüzden
  pratikte önemlidir. Veri seti farklı kalitede politikaların karışımını içerir (rastgele → uzman). Ayrıca
  **öğrenilen her politika simülatörde birebir değerlendirilebilir**. Bu, gerçek veri setlerinde mümkün olmayan bir
  avantajdır ve **off-policy değerlendirme (OPE)** yöntemlerinin tahminlerini gerçek değerle kıyaslamaya imkân verir.
- **Önerilen yöntemler:** Davranış klonlama (BC), CQL, IQL, BCQ, Decision Transformer; veri kalitesi ablasyonu
  (yalnızca `random` / yalnızca uzman / karışık); OPE (FQE, önem örneklemesi).
- **Referans sonuç (BC, `restore_decoys` verisiyle, ortamda 500 bölüm):**

| Saldırgan | Öğretmen (`restore_decoys`) | BC politikası |
|---|---|---|
| bline | -55.6 | -56.4 |
| meander | -85.0 | -87.8 |

  BC öğretmenini neredeyse birebir taklit ediyor. Açık soru: offline RL, veri setindeki en iyi politikayı **geçebilir mi**?
  Bu, karışık kaliteli veriyle (ör. `random` + `react_restore`) test edilebilir.

### 6.5 Dünya modeli / dinamik öğrenme

- **Problem:** (gözlem, gerçek durum, aksiyon) → (sonraki gözlem, sonraki gerçek durum, ödül) tahmini.
  Yükleyici: `dynamics_view`.
- **Neden ilginç:** Model tabanlı RL ve planlama için temel bileşendir. Gerçek durum mevcut olduğundan "gizli durum
  geçişini" ve "gözlem modelini" ayrı ayrı öğrenmek mümkündür. Öğrenilen bir model simülatörün yerine geçebilir
  (sürrogat simülatör).
- **Metrikler:** Bir adımlık doğruluk, çok adımlı açılımda (rollout) hata birikimi, öğrenilen modelde eğitilmiş
  politikanın gerçek simülatördeki performansı.

### 6.6 Dağılım kayması ve genelleme

- **Hazır kaymalar:** (1) `ood_test`: eğitimde görülmeyen savunmacı politikası. (2) Saldırgan kayması: yalnızca
  B-line ile eğitip Meander'da test etme. (3) Simülatör sürümü: `--remove-bugs false` ile üretilen ikinci veri seti.
  (4) Simülatörler arası: aynı senaryonun tam CAGE 2 simülatöründen üretilebilecek küçük bir test seti (henüz üretilmedi).
- **Bulgu:** Tespit görevinde OOD düşüşü ılımlıyken, erken uyarı görevinde çok belirgin (§6.1, §6.2). Literatürde
  CAGE ajanlarının yeni ağlara ve saldırganlara genellemede zorlandığı da gösterilmiştir (Wolk vd., 2022; §9.3).
- **Çalışılabilecek yöntemler:** Alan genellemesi (domain generalization), politikadan bağımsız temsil öğrenme,
  belirsizlik tahmini ve OOD tespiti (model ne zaman yanıldığını bilebilir mi?).

### 6.7 Tablo verisi ve yorumlanabilirlik

`tabular.csv.gz` her adımı okunabilir sütunlarla verir: politika adları, aksiyon adları (`exploit_ent1` gibi), başarı
bayrakları, ödül ve kümülatif ödül, ele geçirilmiş host sayıları, `opserv` impact durumu, host başına uyarı ve gerçek
etiket. Gradient boosting (XGBoost / LightGBM) ile tespit, SHAP ile hangi gözlem özelliklerinin karar verdiğinin
analizi ve keşifsel veri analizi (EDA) için uygundur. Lisans düzeyindeki projeler için iyi bir başlangıç noktasıdır.

### 6.8 Etiket verimliliği ve yarı denetimli öğrenme

Etiketler burada bedava olsa da gerçek dünyada pahalıdır. Etiketlerin yalnızca küçük bir kısmını kullanarak
(ör. bölümlerin %1'i) yarı denetimli, öz-denetimli (self-supervised) ön eğitim veya aktif öğrenme yöntemleri
denenebilir. Tam etiketli sonuçla kıyaslanarak "kaç etiket yeterli?" sorusu ölçülebilir.

### Önerilen çalışma soruları

1. Dizi modelleri (GRU/Transformer), pencereli MLP'ye göre tespitte ve özellikle restore sonrası kaybolan uyarılarda
   ne kadar kazanç sağlar?
2. Ağ topolojisini kullanan bir grafik sinir ağı, host başına tespitte düz MLP'yi geçer mi?
3. Erken uyarı modelinin OOD'daki çöküşü hangi yöntemlerle (veri artırma, politika rastgeleleştirme, alan genellemesi)
   azaltılabilir?
4. Offline RL, karışık kaliteli veriden veri setindeki en iyi savunmacıdan daha iyi bir politika çıkarabilir mi?
5. OPE yöntemlerinin tahminleri, simülatörde ölçülen gerçek politika değerine ne kadar yakın?

---

## 7. Doğrulama

Veri setinin doğru üretildiğini göstermek için yapılan kontroller (`validate.py`):

**7.1 Kaynak koddaki ödül hatası (düzeltildi).** MiniCAGE'in `SimplifiedCAGE.step()` fonksiyonu ödülü iki kez
hesaplıyor ve hesaplama fonksiyonu ödül sözlüğünü yerinde değiştirdiği için durum cezası iki kez düşülüyor.
Meander × Sleep için depodaki kod **-2119.8**, düzeltilmiş sürüm **-1059.9** veriyor (yayımlanmış değer −1067).
Veri seti düzeltilmiş sürümle (`experiments/dataset/env.py::FixedCAGE`) üretildi; orijinal depo dosyaları değiştirilmedi.

**7.2 Saldırgan ajanı.** Paketin dışa açtığı B-line ajanı (`test_agent.B_line_minimal`) birkaç adım sonra aynı
aksiyonda takılı kalıyor (B-line × Sleep ≈ −10, yayımlanmış değer −1141). Bunun yerine depodaki çalışan sürüm
(`red_bline_agent.B_line_minimal`) kullanıldı.

**7.3 Yayımlanmış sonuçlarla tutarlılık** (`remove_bugs=False`, 500 bölüm × 100 adım, MiniCAGE README tablosu):

| Red | Blue | Yayımlanmış | Bu çalışma (ort. ± SE) | Fark |
|---|---|---|---|---|
| bline | react_restore | -156 | -89.6 ± 2.3 | %42.6 |
| bline | restore_decoys | -68 | -47.9 ± 0.3 | %29.5 |
| bline | sleep | -1141 | -1134.6 ± 0.8 | %0.6 |
| meander | react_restore | -68 | -93.5 ± 1.8 | %37.5 |
| meander | restore_decoys | -63 | -66.2 ± 1.3 | %5.1 |
| meander | sleep | -1067 | -1061.4 ± 1.1 | %0.5 |

Savunmasız (`sleep`) senaryolarda ve Meander × Restore-Decoys'ta sonuçlar yayımlanmış değerlerle %0.5–5 içinde
örtüşüyor; ortam dinamiği ve ödül hesabı doğru çalışıyor. React-Restore satırları ise belirgin şekilde farklı. Depodaki
`React_restore_minimal` uygulaması, yayımlanmış tabloyu üreten uygulamadan farklı görünüyor. Bu yalnızca o savunma
politikasının skorunu etkiler, ortamı ya da etiketleri etkilemez.

**7.4 Etiket tutarlılığı:**

| Kontrol | Beklenen | Ölçülen |
|---|---|---|
| B-line × Sleep bölümlerinin sonunda `opserv`'te root etiketi olan oran | 1.000 | 1.000 |
| `opserv` impact altındayken ödülün ≤ −10 olduğu adım oranı | 1.000 | 1.000 |
| `opserv` impact altındayken `opserv` etiketinin root olduğu adım oranı | 1.000 | 1.000 |
| Başarılı restore sonrası hostun temiz etiketlendiği oran (745 483 restore) | 1.000 | 1.000 |
| `react_restore`: React-Restore dışı aksiyon oranı (beklenen ε·39/53) | 0.000 | 0.000 |
| `react_restore_eps0.1`: React-Restore dışı aksiyon oranı (beklenen ε·39/53) | 0.074 | 0.073 |
| `react_restore_eps0.3`: React-Restore dışı aksiyon oranı (beklenen ε·39/53) | 0.221 | 0.220 |

**7.5 Yeniden üretilebilirlik:** Aynı seed ile yeniden üretilen shard'lar tüm alanlarda birebir aynı: `shard_000.npz` ✓, `shard_023.npz` ✓.

---

## 8. Referans sonuçların deney ayrıntıları

Referans modeller yalnızca verinin öğrenilebilir olduğunu göstermek ve karşılaştırma için bir taban sağlamak amacıyla
eğitildi; hiperparametre araması yapılmadı. Tümü CPU'da, toplam ≈1 dakikada çalışır (`baselines.py`).

| Görev | Model | Eğitim verisi | Eğitim |
|---|---|---|---|
| Tespit, erken uyarı, Red aksiyon tipi | MLP (2 × 256, ReLU) | Train'den rastgele 4000 bölüm (erken uyarıda 6000) | Adam, lr 1e-3, 5 epoch, batch 1024 |
| Saldırgan tanıma | GRU (64 gizli birim) | Tüm train (16 000 bölüm) | Adam, lr 1e-3, 10 epoch, batch 256 |
| Davranış klonlama | MLP (2 × 256) | Train'deki `restore_decoys` bölümleri | 10 epoch; ortamda 500 bölümle değerlendirme |

"Basit referans" sütunları: **çoğunluk sınıfı** (her zaman en sık sınıfı tahmin et), **uyarı kuralı** (gözlemdeki
istismar bayrağı varsa ihlal var de; erken uyarıda `ent2` veya `opserv`'te uyarı varsa impact gelecek de), **şans**.

Tüm sonuçlar ham hâliyle `baselines.json` dosyasındadır.

---

## 9. İlgili çalışmalar ve konumlandırma

*Bu bölüm Ekim 2026 itibarıyla yapılan bir literatür taramasına dayanır; kapsamlı bir derleme değildir.*

### 9.1 Klasik saldırı tespit (IDS) veri setleri

Makine öğrenmesiyle saldırı tespiti çalışmalarında yaygın kullanılan veri setleri ağ trafiği veya sistem loglarından
oluşur:

| Veri seti | İçerik | Bu veri setinden farkı |
|---|---|---|
| NSL-KDD, UNSW-NB15, CIC-IDS2017, CSE-CIC-IDS2018 | Ağ akışı (flow) özellikleri, akış başına saldırı etiketi | Statik; saldırı aşaması ve gizli durum yok; savunmacı aksiyonu yok |
| DARPA TC / OpTC | Kurumsal ağda (OpTC: ~1000 Windows host) sistem olayları, çok aşamalı APT senaryoları [16] | Gerçekçi ama statik; etiketleme kısmi ve yorum gerektirir |
| AIT Log Data Set | Test ortamından sistem logları, satır başına saldırı adımı etiketi [15] | Saldırı adımı etiketi var; etkileşim yok |
| DAPT 2020, Unraveled | Ağ akışı ve sistem logları; APT aşaması etiketleri (DAPT 2020'de kısmi, Unraveled'da yarı sentetik ve kapsamlı) [16, 18] | Etkileşim yok; savunmacı kararları çalışılamaz |

Bu setlerin en yaygın kullanılanlarında **etiket kalitesi** belgelenmiş bir sorundur. Engelen vd., CIC-IDS2017'deki
akışların %25'ten fazlasının (bazı saldırı türlerinde %50'ye varan) anlamsız yapay kalıntılar olduğunu ve etiketleme
hataları bulunduğunu göstermiştir [12]. Benzer hatalar CSE-CIC-IDS2018 için de raporlanmış ve bu hataların tespit
performansını belirgin biçimde değiştirdiği gösterilmiştir [13]. Arp vd., güvenlik alanındaki ML çalışmalarında
örnekleme yanlılığı ve veri sızıntısı gibi hataların yaygın olduğunu belgelemiştir [14].

Bu veri seti bu sorunlara farklı bir açıdan yaklaşır: etiketler simülatörün kendi durumundan geldiği için kesindir,
ve veri etkileşimlidir. Ancak paket veya log içermediğinden **bu setlerle doğrudan kıyaslanamaz** ve onların yerine
geçmez.

### 9.2 Otonom siber savunma: veri seti değil, ortam paylaşılıyor

Otonom siber savunma alanında araştırmacılar veri seti yerine **simülasyon ortamları** (gym) paylaşır: CybORG ve
CAGE Challenge 2/3/4, CybORG++ / MiniCAGE [1], Yawning Titan, PrimAITE, CyberBattleSim, NASim ve Cyberwheel [2, 10].
Sonuçlar genellikle bu ortamlarda eğitilen ajanın bölüm ödülüyle raporlanır.

- **Ortak zemin CAGE-2'dir.** Le & Stadler CAGE-2'yi "savunma stratejilerini öğrenmek ve değerlendirmek için kabul
  görmüş bir benchmark" olarak tanımlar [6]. Kiely vd. CAGE-2 katılımcılarının yöntemlerini sınıflandırmış ve
  hiyerarşik derin RL'nin en başarılı yaklaşım olduğunu bulmuştur [9].
- **Alan parçalıdır.** Hicks vd. (2026) tek bir standart benchmark olmadığını, ortamların ciddi özelleştirme
  gerektirdiğini belirtir. Ayrıca mevcut ortamları üç noktada eleştirir: gerçekte bilinemeyecek bilgiyi gözleme
  koyan "sihirli gözlemler", sabit stratejili saldırganlar ve sıra tabanlı etkileşim. CybORG++ / MiniCAGE'i de bu
  bağlamda anar [2].

**Yörünge veri setleri.** Taramada bu veri setine en yakın çalışma **Trident**'tir (Masukawa vd., 2026): CAGE 4 ve
Cyberwheel ortamlarından 13 000'i aşkın saldırgan-savunmacı yörüngesi [4]. Ancak amacı farklıdır: yörüngeler
doğal dil logu biçimindedir ve LLM tabanlı saldırgan ajanların eğitimi için tasarlanmıştır. Kodu yazım sırasında
yalnızca anonim bir hakem deposunda yer almaktadır. Offline RL ile savunma ajanı eğitimi CyberVAN ortamında da
çalışılmıştır [11]; bu çalışmanın verisinin açık olduğuna dair bir bilgiye ulaşılamamıştır.

**Bulunamayan:** CAGE-2 senaryosundan üretilmiş, adım bazında gizli durum etiketli, tespit, saldırgan modelleme ve
offline RL için hazırlanmış **açık** bir veri seti taramada bulunamamıştır.

### 9.3 Bu veri setindeki görevlerle ilgili çalışmalar

Görevlerin çoğu CAGE-2 üzerinde ayrı ayrı çalışılmıştır, ancak bu çalışmalar kendi verilerini kendileri üretir ve
yayımlamaz:

| Görev (bu belgede) | İlgili çalışma |
|---|---|
| Red aksiyon tahmini (§6.3) | Samaddar vd. (2026), CAGE-2'de Blue gözlemleri ve Blue aksiyonlarından Red'in aksiyonunu taklit öğrenmeyle tahmin eder. B-line ile Meander arasında bölüm içinde geçiş yapan bir saldırgan (RedSwitch) da kullanır [5]. |
| Saldırgan tanıma (§6.3) | CAGE-2'nin kazanan çözümleri önce saldırganın türünü tanıyıp ona özel politika seçer [3, 9]. Görevin pratikte kullanıldığını ve ilk adımlarda kolay çözüldüğünü gösterir. |
| Gizli durum kestirimi / tespit (§6.1) | Le & Stadler (2025) CAGE-2'yi POMDP olarak modeller, parçacık filtresiyle inanç durumu tutar ve liderlikteki CARDIFF çözümünü geçer [6]. Hammar & Li (2025) CAGE-2 üzerinde Bayesçi öğrenmeyle inanç tabanlı müdahale planlaması yapar [7]. |
| Dağılım kayması (§6.6) | Wolk vd. (2022) CAGE ajanlarının görülmemiş ağlarda ve yeni saldırgan stratejilerine karşı belirgin şekilde kötüleştiğini gösterir [8]. Bu veri setinin erken uyarı görevindeki OOD düşüşü (§6.2) aynı yöndedir. |
| Offline RL (§6.4) | CyberVAN ortamında offline RL ile savunma ajanı eğitimi [11]. |

### 9.4 Konumlandırma

**Bu veri seti nedir:** CAGE-2 senaryosundan (MiniCAGE üzerinden) türetilmiş, denetimli ve offline öğrenme görevleri
için hazırlanmış bir benchmark veri setidir. Katkısı, literatürde ayrı ayrı ve yayımlanmamış verilerle çalışılan
görevler (tespit / inanç kestirimi, saldırgan modelleme, offline RL) için **ortak, açık, kesin etiketli ve yeniden
üretilebilir** bir veri tabanı sunmasıdır. Belge biçimi Gebru vd.'nin "Datasheets for Datasets" önerisini izler [17].

**Bu veri seti ne değildir:** Gerçek ağ trafiği içeren bir IDS veri seti değildir ve §9.1'deki setlerle doğrudan
kıyaslanamaz. MiniCAGE, CAGE-2'nin kendisi değil, onun hızlı bir yeniden uygulamasıdır (eşdeğerlik iddiası [1]'e
aittir; §7.3'te React-Restore skorlarında fark gözlenmiştir).

**Yayın düzeyine taşımak için gerekenler:**

1. Saldırgan çeşitliliği: bölüm içinde strateji değiştiren saldırgan (RedSwitch [5]), gürültülü ve öğrenen (PPO)
   saldırganlar.
2. Tam CybORG simülatöründen küçük bir test seti (simülatörler arası transfer).
3. Doğrudan kıyaslamalar: Red aksiyon tahmininde [5], gizli durum kestiriminde parçacık filtresi tabanlı yaklaşımla [6].
4. Referans modellerin birden fazla seed ile koşulması ve varyans raporlanması [2].
5. Zenodo veya Hugging Face üzerinden DOI ile yayımlama; değerlendirmenin [14]'teki kontrol listesine göre yapılması.

### Kaynakça

1. H. Emerson, L. Bates, C. Hicks, V. Mavroudis. *CybORG++: An Enhanced Gym for the Development of Autonomous Cyber Agents.* arXiv:2410.16324, 2024. https://arxiv.org/abs/2410.16324
2. C. Hicks, E. Bates, S. McFadden, I. Symes Thompson vd. *Building Better Environments for Autonomous Cyber Defence.* AI4CNI Workshop @ AAMAS 2026. arXiv:2604.08805. https://arxiv.org/abs/2604.08805
3. C. Hicks, V. Mavroudis. *Autonomous cyber defence: Beyond games?* Turing Technical Report No. 2, 2024. https://doi.org/10.5281/zenodo.10974183
4. R. Masukawa vd. *Trident: How to Break Deep Reinforcement Learning Cyber Defenses (Agentic).* arXiv:2608.04317, 2026. https://arxiv.org/abs/2608.04317
5. A. Samaddar, S. Neema, D. Balasubramanian, X. Koutsoukos. *Learning Red Agent Policy from Observations for Neurosymbolic Autonomous Cyber Agents.* arXiv:2606.18223, 2026. https://arxiv.org/abs/2606.18223
6. D. H. Le, R. Stadler. *Learning Optimal Defender Strategies for CAGE-2 using a POMDP Model.* CNSM 2025. arXiv:2509.06539. https://arxiv.org/abs/2509.06539
7. K. Hammar, T. Li. *Online Incident Response Planning under Model Misspecification through Bayesian Learning and Belief Quantization.* ACM CCS AISec 2025. arXiv:2508.14385. https://arxiv.org/abs/2508.14385
8. M. Wolk vd. *Beyond CAGE: Investigating Generalization of Learned Autonomous Network Defense Policies.* NeurIPS 2022 Workshop on RL for Real Life. arXiv:2211.15557. https://arxiv.org/abs/2211.15557
9. M. Kiely, D. Bowman, M. Standen, C. Moir. *On Autonomous Agents in a Cyber Defence Environment.* 2nd International Workshop on Adaptive Cyber Defence, 2023. arXiv:2309.07388. https://arxiv.org/abs/2309.07388
10. G. Palmer vd. *Deep Reinforcement Learning for Autonomous Cyber Defence: A Survey.* arXiv:2310.07745. https://arxiv.org/abs/2310.07745
11. *Offline Reinforcement Learning for Autonomous Cyber Defense Agents.* IEEE Xplore, doküman 10838812. https://ieeexplore.ieee.org/document/10838812/
12. G. Engelen, V. Rimmer, W. Joosen. *Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study.* IEEE Security and Privacy Workshops, 2021. https://lirias.kuleuven.be/retrieve/713070
13. *Errors in the CICIDS2017 dataset and the significant differences in detection performances it makes.* 2022. https://hal.archives-ouvertes.fr/hal-03775466 ; *Error Prevalence in NIDS datasets: A Case Study on CIC-IDS-2017 and CSE-CIC-IDS-2018.* https://archive.cps-vo.org/node/97449
14. D. Arp vd. *Dos and Don'ts of Machine Learning in Computer Security.* USENIX Security 2022. arXiv:2010.09470. https://arxiv.org/abs/2010.09470
15. *AIT Log Data Set.* Austrian Institute of Technology, Zenodo. https://zenodo.org/records/5789064
16. *SAGA: Synthetic Audit Log Generation for APT Campaigns* (mevcut APT veri setlerinin karşılaştırması). arXiv:2411.13138. https://arxiv.org/abs/2411.13138
17. T. Gebru vd. *Datasheets for Datasets.* Communications of the ACM, 64(12), 2021.
18. S. Myneni vd. *DAPT 2020 — Constructing a Benchmark Dataset for Advanced Persistent Threats*, 2020; *Unraveled — A semi-synthetic dataset for Advanced Persistent Threats*, Computer Networks, 2023.

---

## 10. Sınırlamalar ve dikkat edilmesi gerekenler

- **Simülasyon verisidir.** MiniCAGE, CAGE 2 senaryosunun soyut bir modelidir: paket, log veya ağ trafiği yoktur;
  sonuçlar doğrudan gerçek ağlara genellenemez. Bu tür ortamlara yöneltilen eleştiriler (sihirli gözlemler, sabit
  saldırganlar, sıra tabanlı etkileşim) bu veri seti için de geçerlidir (§9.2).
- **Sabit ve küçük topoloji** (13 host, 3 alt ağ).
- **Saldırgan çeşitliliği düşük:** iki deterministik betik. Saldırgan modelleme görevleri bu yüzden kolay (§6.3).
- **Savunmacılar betik tabanlı veya rastgeledir;** eğitilmiş (ör. PPO) bir uzman politika henüz yok.
- **Zamansal bağımlılık:** Aynı bölümün adımları birbirinden bağımsız değildir. Değerlendirmede bölüm düzeyinde
  bölmeye (`splits.npz`) uyulmalı, adımlar rastgele karıştırılarak bölünmemelidir.
- **Sınıf dengesizliği:** Bkz. §5.2.
- **Green ajanının taramaları** gözlemde gürültü olarak vardır ama ayrı bir alan olarak kaydedilmez.
- **Kaynak koddaki iki hata** bu veri setinde düzeltilmiştir (§7.1, §7.2). Orijinal depoyu doğrudan kullanan
  çalışmalarla sayısal karşılaştırma yaparken bu dikkate alınmalıdır.

**Etik:** Veri tamamen sentetiktir; kişisel veri, gerçek sistem veya gerçek saldırı kaydı içermez. Amaç savunma
araştırmasıdır.

---

## 11. Genişletme olanakları

Üretim hattı parametriktir; aşağıdakiler birkaç satır kod ve birkaç dakikalık üretimle eklenebilir:

- **Gürültülü / karışık saldırganlar** (ε-B-line, bölüm içinde strateji değiştiren saldırgan; bkz. RedSwitch, §9.3) → saldırgan modelleme
  görevlerini zorlaştırır.
- **Eğitilmiş PPO savunmacısı** → offline RL için "uzman" veri.
- **`remove_bugs=False` sürümü** → simülatör sürümleri arası dağılım kayması.
- **Tam CAGE 2 simülatöründen küçük bir test seti** → simülasyondan simülasyona transfer.
- **Farklı bölüm uzunlukları** (30 / 50 / 100 adım), CAGE Challenge'daki değerlendirme protokolüne uyum için.

---

## 12. Yeniden üretim

`<repo>` dizininde, CybORG++ deposu ve `venv_cyborg` ortamı ile:

```bash
venv_cyborg/bin/python -m experiments.dataset.generate --episodes-per-config 2000   # ≈23 s
venv_cyborg/bin/python -c "from experiments.dataset import loaders as L; L.export_tabular('experiments/data/minicage_v1')"
venv_cyborg/bin/python -m experiments.dataset.baselines                              # ≈1 dk
venv_cyborg/bin/python -m experiments.dataset.validate                               # kontroller + bu belge
```

Bağımlılıklar: Python 3.12.3, NumPy 2.5.3, PyTorch (yalnızca referans modeller için).

---

## 13. Lisans ve atıf

Simülatör CybORG++ MIT lisansı ile dağıtılır. Veri setini kullanan çalışmalar simülatöre atıf yapmalıdır:

```bibtex
@article{emerson2024cyborg++,
  title   = {Cyborg++: An enhanced gym for the development of autonomous cyber agents},
  author  = {Emerson, Harry and Bates, Liz and Hicks, Chris and Mavroudis, Vasilios},
  journal = {arXiv preprint arXiv:2410.16324},
  year    = {2024}
}
```

---

## Ek: Terimler

| Terim | Anlamı |
|---|---|
| Bölüm (episode) | Ortamın sıfırlanmasından itibaren 100 adımlık tek bir saldırı-savunma oyunu |
| Exploit (istismar) | Bir servisteki açığı kullanarak hosta kullanıcı düzeyinde erişim kazanma |
| Escalate | Kullanıcı erişimini root (yönetici) erişimine yükseltme |
| Impact | Kritik sunucuda hizmet kesintisi yaratma |
| Decoy (tuzak) | Saldırganın istismar girişimini boşa çıkaran sahte servis |
| Restore | Hostu temiz bir yedekten geri yükleme |
| POMDP | Kısmen gözlemlenebilir Markov karar süreci: ajan gerçek durumu değil, ondan türeyen gözlemi görür |
| OOD | Dağılım dışı (out-of-distribution): eğitim verisinde temsil edilmeyen koşullar |
| D4RL | Offline RL için yaygın kullanılan veri seti biçimi ve kıyaslama paketi |
| OPE | Off-policy değerlendirme: bir politikanın değerini başka politikaların verisinden tahmin etme |
