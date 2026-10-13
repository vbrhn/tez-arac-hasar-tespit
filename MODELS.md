***Türkçe** · [English](MODELS.en.md)*

# Model Kartı

Otomatik araç hasar tespit sistemi — eğitilmiş model ağırlıkları, eğitim
koşulları, değerlendirme sonuçları ve sınırlılıklar.

Son güncelleme: 2026-10-13

---

## 1. Dağıtılan Modeller

| Dosya | Rol | Mimari | Boyut |
|---|---|---|---|
| `model1_parca_konum_yolo11m-seg.pt` | **Model 1** — araç parçası bölütleme | YOLOv11m-seg | 45 MB |
| `model2_hasar_nihai_yolo11l-seg_fp32.pt` | **Model 2** — hasar bölütleme ⭐ | YOLOv11l-seg | 56 MB |
| `model2_hasar_v1_sadece_kendi_veri_yolo11m-seg.pt` | ablasyon — V1 referansı | YOLOv11m-seg | 45 MB |
| `model2_hasar_asama1_fp16_yolo11l-seg.pt` | ara kontrol noktası — fp32 devamının başlangıcı | YOLOv11l-seg | 222 MB |

Üretim kullanımı için gereken iki dosya **Model 1** ve **Model 2**'dir.
Diğer ikisi yalnızca yeniden üretim ve ablasyon doğrulaması içindir.

`model2_hasar_asama1_fp16_yolo11l-seg.pt` optimizatör durumunu da içerdiği
için büyüktür (devam eğitimi bu dosyadan başlatılmıştır). Nihai model
`strip_optimizer` sonrası 56 MB'dir.

### İndirme

Ağırlıklar GitHub dosya boyutu sınırları nedeniyle depoda tutulmamaktadır.

| Kaynak | Bağlantı |
|---|---|
| Google Drive | `<paylaşım bağlantısı buraya>` |
| Zenodo (DOI) | `<DOI buraya>` |

---

## 2. Model 2 — Hasar Bölütleme (nihai)

### 2.1 Amaç

Araç dış yüzeyi görüntüsünden hasar bölgelerini piksel düzeyinde çıkarır ve
üç sınıfa ayırır.

| Sınıf kimliği | Ad | Açıklama |
|---|---|---|
| 0 | Çizik | Boya yüzeyinde çizgisel iz |
| 1 | Göçük | Sac yüzeyde şekil bozukluğu |
| 2 | Kırık | Kopma, çatlama, parça kaybı |

### 2.2 Eğitim verisi

| | Değer |
|---|---|
| Eğitim görüntüsü | 15.490 |
| Doğrulama görüntüsü | — (bölme oranı %70/%15/%15) |
| Test görüntüsü | 2.214 (4.458 örnek) |
| Özgün (kendi) veri oranı | %79 |
| Harici kaynak | Roboflow `cocotoyolo-cmnax/damage_2-w8sdi` (yalnızca Göçük sınıfı) |

Veri hazırlığında uygulananlar:

- **Algısal karma ile yineleme eleme** (dHash, Hamming mesafesi ≤ 3)
- **Örtüşmeli kutucuklama** — 2×2 ızgara, %20 örtüşme, Shapely ile poligon
  kırpma; 5.163 → 24.568 örnek (4,8 kat artış)
- **Aşırı örnekleme** — azınlık sınıfları için `OS` önekiyle diske kopyalama

Harici veri setindeki `scratch` / `crack` sınıf tanımları kendi etiket
setimizdeki `Cizik` / `Kirik` ile ters düştüğü için bu iki sınıf harici
kaynaktan **alınmamış**, yalnızca `dent` → `Gocuk` eşlemesi kullanılmıştır.

### 2.3 Eğitim koşulları

İki aşamalı eğitim. İlk aşama fp16 ile 150 devire kadar planlanmış, 83. devirde
maske kaybında NaN taşması oluşmuş, eğitim 82. devrin `best.pt` dosyasından
fp32 ile sürdürülmüştür.

| Parametre | Aşama 1 (fp16) | Aşama 2 (fp32) |
|---|---|---|
| Başlangıç ağırlığı | `yolo11l-seg.pt` (COCO) | aşama 1 `best.pt` (devir 82) |
| Devir | 150 (83'te kesildi) | 70 |
| `amp` | `True` | **`False`** |
| `lr0` | 0.001 | 0.0004 |
| `lrf` | 0.01 | 0.01 |
| Optimizatör | AdamW | AdamW |
| `cos_lr` | `True` | `True` |
| `warmup_epochs` | 3.0 | 2.0 |
| Görüntü boyutu | 960 | 960 |
| Yığın boyutu | 6 | 6 |
| `patience` | 35 | 35 |
| `close_mosaic` | 20 | 20 |

Veri çoğaltma (her iki aşamada aynı):

```
hsv_h=0.015   hsv_s=0.7     hsv_v=0.5
fliplr=0.5    flipud=0.0    degrees=5.0
mosaic=0.5    copy_paste=0.3
scale=0.4     translate=0.1 erasing=0.2
```

Kayıp ağırlıkları: `box=8.0`, `cls=0.5`, `dfl=1.5`

**fp32'ye geçiş gerekçesi.** fp16'nın alt sınırı ~10⁻⁸, fp32'ninki ~10⁻³⁸'dir.
Maske kaybı hesabında çok küçük ara değerler oluştuğunda fp16 taşarak NaN
üretmektedir. `amp=False` bu riski ortadan kaldırır, karşılığında eğitim
yaklaşık 1,6 kat yavaşlar.

Devam eğitiminde öğrenme oranının 0.001 yerine 0.0004 seçilmesi, önceden
öğrenilmiş temsillerin büyük gradyan adımlarıyla bozulmasını (catastrophic
forgetting) önlemek içindir.

**Not.** Devam eğitiminin ilk devrinde maske mAP@50 sıfır ölçülmüştür; aşama 1
`best.pt` dosyasındaki maske başlığı NaN taşmasından etkilenmiştir. Maske
başlığı sıfırdan yeniden öğrenmiş, 33. devirde önceki en iyi değeri geçmiştir.

| Toplam eğitim süresi | ~60 saat (Google Colab Pro+) |
|---|---|
| Parametre sayısı | 27.586.905 |

### 2.4 Değerlendirme — bağımsız test kümesi

2.214 görüntü / 4.458 örnek.

| Metrik | Kutu | Maske |
|---|---|---|
| Kesinlik | 0.7475 | 0.7544 |
| Duyarlılık | 0.6734 | 0.6563 |
| mAP@50 | 0.7422 | **0.7238** |
| mAP@50-95 | 0.5532 | 0.4743 |

Sınıf bazında maske mAP@50:

| Sınıf | mAP@50 |
|---|---|
| Çizik | 0.6381 |
| Göçük | 0.7275 |
| Kırık | 0.8058 |

Karışıklık matrisi (güven 0.25, IoU 0.45) köşegen değerleri: Çizik 0.637,
Göçük 0.737, Kırık 0.785. Arka planın hatalı olarak hasar sayıldığı durumlar:
403 + 622 + 311 = 1.336.

Çıkarım hızı (960 px): 1.23 ms ön işleme + 6.46 ms çıkarım + 1.15 ms son
işleme = **8.84 ms/görüntü**.

### 2.5 Çapraz doğrulama

3 katlı K-fold, maske mAP@50:

| Kat | mAP@50 |
|---|---|
| 1 | 0.6358 |
| 2 | 0.6479 |
| 3 | 0.6335 |
| **Ortalama** | **0.6391 ± 0.0078** |

Değişim katsayısı %1.22 — katlar arası tutarlılık yüksektir. K-fold değerleri
nihai modelden düşüktür; çapraz doğrulama daha küçük veri havuzu ve daha az
devir ile yürütülmüştür.

---

## 3. Model 1 — Parça/Konum Bölütleme

| | |
|---|---|
| Mimari | YOLOv11m-seg |
| Veri seti | Carparts-Seg |
| Görüntü boyutu | 960 px |
| Rol | Hasarın hangi araç parçasında olduğunu belirlemek |

Doğrulama kümesi:

| Metrik | Kutu | Maske |
|---|---|---|
| Kesinlik | 0.6568 | 0.6670 |
| Duyarlılık | 0.8249 | 0.8410 |
| mAP@50 | 0.7042 | 0.7195 |
| mAP@50-95 | 0.6145 | 0.5967 |

Bu model hasar tespitinden bağımsızdır ve yeniden eğitilmemiştir. Duyarlılığın
kesinlikten belirgin biçimde yüksek olması füzyon için istenen davranıştır:
parça kaçırmak, fazladan parça önermekten daha maliyetlidir.

---

## 4. Füzyon

Hasar maskesi `H`, parça maskesi `P` olmak üzere örtüşme oranı:

```
ortusme(H, P) = alan(H ∩ P) / alan(H)
```

`ortusme ≥ 0.38` olan her parça, o hasara atanır. Eşik, doğrulama kümesi
üzerinde parça atama doğruluğu ile fazla atama sayısı arasındaki dengeye göre
seçilmiştir.

Bir hasar birden çok parçaya atanabilir. Hiçbir parça eşiği aşmazsa hasar
`parca=None` ile raporlanır.

---

## 5. Ablasyon

| Sürüm | Değişiklik | Maske mAP@50 |
|---|---|---|
| V0 | Mimari seçimi — YOLOv11m-seg, 640 px, 60 devir | 0.4325 |
| V1 | Veri temizliği, yineleme eleme | tezde Tablo 7 |
| V2 | 960 px + örtüşmeli kutucuklama | tezde Tablo 7 |
| V3 | YOLOv11l-seg, fp32 devam eğitimi | **0.7238** |

V0 için kesinlik 0.5290, duyarlılık 0.4513'tür. V0 koşusunda sınıf bazında
metrikler kaydedilmediği için (`sinif_mAP50` alanı o dönemde yazılmıyordu)
sınıf kırılımı mevcut değildir.

Ara sürümlerin sayısal değerleri tezin Tablo 7'sinde verilmiştir;
`metrik_toplayici.ipynb` Drive'daki tüm metrik JSON dosyalarını tek bir zip'te
toplar.

---

## 6. Sınırlılıklar

### 6.1 Sürümler arası karşılaştırmanın geçerliliği

Her sürümde veri bölme hücresi yeniden çalıştırılmıştır. `seed=42` sabit
olmasına rağmen, bölünen havuzun içeriği sürümler arasında değiştiği için
üretilen eğitim/doğrulama/test kümeleri birebir aynı değildir. Bu, iki yönlü
bir yanlılık doğurur: bir sürümün test kümesinde yer alan bir görüntü, başka
bir sürümün eğitim kümesinde bulunabilir; ayrıca kümelerin sınıf dağılımları
farklılaşır.

Sonuç: ablasyon tablosundaki sürüm karşılaştırmaları **göstergesel**dir, sabit
bir test kümesi üzerinde yapılmış kontrollü karşılaştırma değildir. Nihai
modelin (V3) kendi test kümesi üzerindeki değerleri geçerlidir.

Bu durum geriye dönük düzeltilememiştir; tüm sürümlerin aynı bölme ile yeniden
eğitilmesi 60 saati aşan ek hesaplama gerektirmektedir.

### 6.2 Alan kayması

Üç farklı harici kaynaktan veri eklendiğinde başarım 0.6894'ten 0.3319'a
düşmüştür. Nedenleri: kaynaklar arasında etiketleme kuralı farkları, sınıf
tanımı uyuşmazlıkları ve görüntü edinim koşullarındaki farklılıklar. Nihai
modelde yalnızca tek bir harici kaynak ve yalnızca tek bir sınıf
(`dent` → `Gocuk`) kullanılmıştır. Deney `05_alan_kaymasi_denemesi.ipynb`
dosyasındadır.

### 6.3 Sınıf bazında başarım farkı

Çizik sınıfı en düşük başarıma sahiptir (0.6381). İnce çizikler düşük
çözünürlükte ve zayıf aydınlatmada kaybolmaktadır. Kırık sınıfı en yüksek
başarımı verir (0.8058); kırıklar görsel olarak en belirgin hasar tipidir.

### 6.4 Arka plan yanlış pozitifleri

Test kümesinde 1.336 örnekte arka plan hasar olarak işaretlenmiştir. Yansımalar,
gölgeler ve kir lekeleri başlıca kaynaklardır. Sigorta uygulamasında bu,
insan eksperin doğrulaması gerektiği anlamına gelir; sistem karar destek
amaçlıdır, otonom karar verici değildir.

### 6.5 Kapsam dışı durumlar

- İç mekân hasarı (döşeme, pano) kapsam dışıdır.
- Mekanik ve yapısal hasar görüntüden tespit edilememektedir.
- Hasar şiddeti (onarım maliyeti) tahmini yapılmamaktadır; yalnızca hasar tipi
  ve konumu üretilir.
- Model Türkiye'deki araç parkı ve ekspertiz fotoğraflama alışkanlıkları
  üzerinden eğitilmiştir; başka coğrafyalarda başarım doğrulanmamıştır.

### 6.6 Hesaplama

Çıkarım hızı Colab GPU üzerinde ölçülmüştür. CPU üzerinde ve mobil cihazlarda
gerçek zamanlı çalışma doğrulanmamıştır.

---

## 7. Etik ve Kullanım

Sistem sigorta ekspertiz sürecinde **karar destek** amacıyla tasarlanmıştır.
Hasar tazminatı kararlarının otomatik olarak verilmesi amaçlanmamıştır;
çıktılar yetkili eksper tarafından doğrulanmalıdır.

Eğitim verisindeki görüntüler araç dış yüzeyine aittir; plaka ve kişi görüntüsü
içeren kareler veri hazırlığında ayıklanmıştır.

---

## 8. Yeniden Üretim

| Adım | Dosya |
|---|---|
| Mimari seçimi (V0) | `notebooks/01_mimari_karsilastirmasi.ipynb` |
| Veri temizliği + kutucuklama (V1, V2) | `notebooks/02_cozunurluk_ve_kutucuklama.ipynb` |
| Nihai model (V3) | `notebooks/03_nihai_model.ipynb` |
| Test değerlendirmesi | `notebooks/04_analiz_ve_degerlendirme.ipynb` |
| Alan kayması deneyi | `notebooks/05_alan_kaymasi_denemesi.ipynb` |
| Metrik toplama | `metrik_toplayici.ipynb` |

Veri seti Roboflow üzerindedir; erişim için `ROBOFLOW_API_KEY` ortam değişkeni
gereklidir.
