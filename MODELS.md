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
| Google Drive | https://drive.google.com/file/d/1v6NTyWzOUKCavj1FhO_DHnj-HExDIn32/view?usp=sharing |


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
| Eğitim kutucuğu | 15.490 |
| Doğrulama kutucuğu | 4.426 |
| Test kutucuğu | 2.214 (4.458 hasar örneği) |
| Bölme oranı | %70 / %20 / %10 (`seed=42`), kutucuklamadan sonra |
| Özgün (kendi) veri oranı | %79 |
| Harici kaynak | Roboflow `cocotoyolo-cmnax/damage_2-w8sdi` (yalnızca Göçük sınıfı) |

Veri hazırlığında uygulananlar:

- **Algısal karma ile yineleme eleme** (dHash, Hamming mesafesi ≤ 3)
- **Örtüşmeli kutucuklama** — 2×2 ızgara; kutucuk boyutu ızgara adımının %20
  fazlası (1024 px'lik görüntüde 614 px kutucuk, komşular arası ~204 px ≈ %33
  örtüşme). Shapely ile poligon kırpma; özgün alanının %30'undan azı kalan
  poligonlar atılır. 7.302 kaynak görüntüden 23.863 kutucuk.
- **Küçük anotasyon süzgeci** — görüntü alanının %0,04'ünden küçük poligonlar
  (Shoelace alanı) çıkarılır.
- **Hasarsız kutucuk oranı** — %25 (ön deneylerde %10)
- **Aşırı örnekleme** — azınlık sınıfları için `OS` önekiyle diske kopyalama

Harici veri setindeki `scratch` / `crack` sınıf tanımları kendi etiket
setimizdeki `Cizik` / `Kirik` ile ters düştüğü için bu iki sınıf harici
kaynaktan **alınmamış**, yalnızca `dent` → `Gocuk` eşlemesi kullanılmıştır.

### 2.3 Eğitim koşulları

İki aşamalı eğitim. İlk aşama fp16 ile 150 devir olarak planlanmış, 83. devirde
maske kaybı tanımsız (NaN) hâle gelmiş, eğitim 82. devrin `best.pt` dosyasından
fp32 ile 70 devir daha sürdürülmüştür (toplam 152 devir).

| Parametre | Aşama 1 (fp16) | Aşama 2 (fp32) |
|---|---|---|
| Başlangıç ağırlığı | `yolo11l-seg.pt` (COCO) | aşama 1 `best.pt` (devir 82) |
| Devir | 1–82 (150 planlandı, 83'te kesildi) | 83–152 (70 devir) |
| `amp` | `True` | **`False`** |
| `lr0` | 0.001 | 0.0004 |
| `lrf` | 0.01 | 0.01 |
| Optimizatör | AdamW | AdamW |
| `cos_lr` | `True` | `True` |
| `warmup_epochs` | 5.0 | 2.0 |
| Görüntü boyutu | 960 | 960 |
| Yığın boyutu | 6 | 6 |
| `patience` | 45 | 35 |
| `close_mosaic` | 20 (devreye girmedi) | 20 |

Veri çoğaltma (her iki aşamada aynı):

```
hsv_h=0.015   hsv_s=0.7     hsv_v=0.5
fliplr=0.5    flipud=0.0    degrees=5.0
mosaic=0.5    copy_paste=0.3
scale=0.4     translate=0.1 erasing=0.2
```

Kayıp ağırlıkları: `box=8.0`, `cls=0.5`, `dfl=1.5`

**fp32'ye geçiş gerekçesi.** fp16'da temsil edilebilen en küçük pozitif değer
~6×10⁻⁸, fp32'de ~10⁻⁴⁵'tir. Maske kaybı hesabındaki çok küçük ara değerler
fp16'da sıfıra yuvarlandığında (alt taşma) ardından gelen bölme tanımsız
hâle gelmekte ve NaN üretmektedir. `amp=False` bu riski ortadan kaldırır,
karşılığında eğitim yavaşlar.

Devam eğitiminde öğrenme oranının 0.001 yerine 0.0004 seçilmesi, önceden
öğrenilmiş temsillerin büyük gradyan adımlarıyla bozulmasını (catastrophic
forgetting) önlemek içindir.

**Not.** Devam eğitiminin (`hasar_yolo11l-seg_v6_fp32/results.csv`) ilk
devrinde doğrulama maske mAP@50 0.544, son (70.) devrinde 0.723'tür.

| Toplam eğitim süresi | ~60 saat (Google Colab Pro+) |
|---|---|
| Parametre sayısı | 27.586.905 |

### 2.4 Değerlendirme — test kümesi

2.214 kutucuk / 4.458 hasar örneği. Test kümesi kutucuk düzeyinde ayrıldığı için
görüntü düzeyinde tam bağımsız değildir (bkz. 6.1).

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
| Eğitim çözünürlüğü | 640 px |
| Sınıflar | 23 Carparts-Seg sınıfı; raporda Türkçe parça adlarına eşlenir |
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

Hasar, en yüksek örtüşmeyi veren parçaya atanır. Bu oran operasyonel eşik
olan 0.10'un altında kalırsa hasar `parca=None` (konum belirsiz) ile raporlanır.
IoU yerine bu oranın kullanılmasının nedeni ölçek farkıdır: parça on binlerce
piksel kaplarken hasar birkaç bin pikseldir; doğru eşleşmede bile IoU ~0.02
düzeyinde kalır.

Birleşik skor = hasar güveni × parça güveni. Parça ataması yapılamayan
tespitlerde skor 0,5 ile çarpılır. Bu değer kalibre edilmiş bir olasılık değil,
iki modelin uyumunu yansıtan operasyonel bir skordur.

---

## 5. Ablasyon

| Sürüm | Değişiklik | Kutu mAP@50 | Maske mAP@50 |
|---|---|---|---|
| V0 | Mimari seçimi — YOLOv11m-seg, 640 px, 60 devir, kutucuklama yok | 0.4768 | 0.4325 |
| V1 | 960 px + 2×2 örtüşmeli kutucuklama, 45 devir | 0.6152 | 0.5918 |
| V2 | 120 devir + hasarsız kutucuk oranı %10 → %25 | 0.7101 | 0.6894 |
| V3 | Harici veri (yalnızca Göçük), sınıf tanımı düzeltmesi, YOLOv11l-seg, fp32 | 0.7422 | **0.7238** |

V2→V3 geçişinde veri, sınıf kümesi, mimari ve sayısal hassasiyet birlikte
değiştiği için bu geçişteki kazanım tek bir etkene atfedilemez.

V0 için kesinlik 0.5290, duyarlılık 0.4513'tür. V0 koşusunda sınıf bazında
metrikler kaydedilmediği için (`sinif_mAP50` alanı o dönemde yazılmıyordu)
sınıf kırılımı mevcut değildir.

Tablo tezdeki Tablo 7 ile aynıdır;
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

Ayrıca bölme kutucuklamadan sonra yapıldığı için bir test kutucuğunun aynı
kaynak görüntüden türeyen komşuları eğitim veya doğrulama kümesinde
bulunabilir; test kümesi görüntü düzeyinde tam bağımsızlık sağlamaz.

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

Test kümesinde (güven 0.25, IoU 0.45) 1.336 tespit hasarsız bölgede üretilmiştir. Yansımalar,
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
