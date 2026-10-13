# Sigorta Sektöründe Derin Öğrenme Tabanlı Otomatik Araç Hasar Tespit Sistemi

İki aşamalı YOLOv11-seg mimarisi ile araç dış yüzeyindeki hasarların otomatik
tespiti, sınıflandırılması ve konumlandırılması.

Bu depo, Üsküdar Üniversitesi Fen Bilimleri Enstitüsü Yapay Zekâ Mühendisliği
Yüksek Lisans Programı kapsamında hazırlanan tez çalışmasının kodlarını,
eğitilmiş model ağırlıklarını ve değerlendirme metriklerini içerir.

| | |
|---|---|
| **Yazar** | Vesalettin Burhan Özdemir (224329044) |
| **Kurum** | Üsküdar Üniversitesi, Fen Bilimleri Enstitüsü |
| **Program** | Yapay Zekâ Mühendisliği (YL) |
| **Yıl** | 2026 |

---

## 1. Sistem Mimarisi

Sistem iki bağımsız örnek bölütleme (instance segmentation) modelinin
füzyonundan oluşur:

```
                    ┌─────────────────────────┐
   Giriş görüntüsü ─┤                         │
                    │  Model 1 — Parça/Konum  │──► kaput, kapı, tampon, far …
                    │  (YOLOv11m-seg)         │
                    └─────────────────────────┘
                                 │
                                 │  örtüşme oranı füzyonu (eşik = 0,38)
                                 │
                    ┌─────────────────────────┐
   Giriş görüntüsü ─┤                         │
                    │  Model 2 — Hasar        │──► Çizik, Göçük, Kırık
                    │  (YOLOv11l-seg)         │
                    └─────────────────────────┘
                                 │
                                 ▼
                 "Sol ön kapıda göçük (%87 güven)"
```

Model 2'nin ürettiği hasar maskesi ile Model 1'in ürettiği parça maskesi
kesiştirilir. Kesişim alanının hasar maskesine oranı 0,38 eşiğini aşarsa hasar
o parçaya atanır. Bir hasar birden fazla parçaya atanabilir (örn. tampon ile
çamurluk arasına yayılan çizik).

---

## 2. Sonuçlar

### 2.1 Nihai model — Model 2 (hasar), bağımsız test kümesi

2.214 görüntü / 4.458 örnek üzerinde ölçülmüştür.

| Metrik | Kutu (Box) | Maske (Mask) |
|---|---|---|
| Kesinlik (Precision) | 0.7475 | 0.7544 |
| Duyarlılık (Recall) | 0.6734 | 0.6563 |
| mAP@50 | 0.7422 | **0.7238** |
| mAP@50-95 | 0.5532 | 0.4743 |

Sınıf bazında maske mAP@50:

| Sınıf | mAP@50 |
|---|---|
| Çizik | 0.6381 |
| Göçük | 0.7275 |
| Kırık | 0.8058 |

Çıkarım hızı: 1.23 ms ön işleme + 6.46 ms çıkarım + 1.15 ms son işleme =
**8.84 ms/görüntü** (Google Colab Pro+ GPU, 960 px).

Doğrulama kümesi ile test kümesi arasındaki fark 0.002'nin altındadır; bu,
modelin doğrulama kümesine aşırı uyum göstermediğini destekler.

### 2.2 Model 1 (parça/konum), doğrulama kümesi

| Metrik | Kutu | Maske |
|---|---|---|
| Kesinlik | 0.6568 | 0.6670 |
| Duyarlılık | 0.8249 | 0.8410 |
| mAP@50 | 0.7042 | 0.7195 |
| mAP@50-95 | 0.6145 | 0.5967 |

### 2.3 Mimari seçiminden nihai modele

| | Mimari seçimi (V0) | Nihai model (V3) |
|---|---|---|
| Model | YOLOv11-Medium | YOLOv11-Large |
| Çözünürlük | 640 px | 960 px |
| Devir | 60 | 150 + 70 (fp32 devam) |
| Maske mAP@50 | 0.4325 | **0.7238** |
| Kesinlik | 0.5290 | 0.7544 |
| Duyarlılık | 0.4513 | 0.6563 |

Ara sürümler (V1: veri temizliği ve yineleme eleme, V2: örtüşmeli kutucuklama)
dâhil tam ablasyon tablosu tezin Tablo 7'sindedir. Metrik dosyalarının
tamamı `metrik_toplayici.ipynb` ile Drive'dan toplanabilir.

### 2.4 Çapraz doğrulama

3 katlı K-fold, maske mAP@50:

| Kat 1 | Kat 2 | Kat 3 | Ortalama |
|---|---|---|---|
| 0.6358 | 0.6479 | 0.6335 | **0.6391 ± 0.0078** (CV %1.22) |

---

## 3. Depo Yapısı

```
.
├── README.md                              bu dosya
├── MODELS.md                              model kartı: tüm metrikler ve sınırlılıklar
├── requirements.txt                       bağımlılıklar
├── predict.py                             tek komutluk çıkarım scripti
├── .gitignore
├── metrik_toplayici.ipynb                 Drive'daki tüm metrik dosyalarını tek zip'te toplar
└── notebooks/
    ├── 01_mimari_karsilastirmasi.ipynb    V0 — beş YOLO varyantının karşılaştırması
    ├── 02_cozunurluk_ve_kutucuklama.ipynb V1+V2 — veri temizliği, 960 px, kutucuklama
    ├── 03_nihai_model.ipynb               V3 — nihai model (YOLOv11l-seg, fp32) ⭐
    ├── 04_analiz_ve_degerlendirme.ipynb   test kümesi değerlendirmesi, gerçek dünya testi
    └── 05_alan_kaymasi_denemesi.ipynb     başarısız genişletme denemesi (negatif sonuç)
```

Model ağırlıkları (`.pt`) boyutları nedeniyle depoya dâhil edilmemiştir;
indirme bağlantıları için **[MODELS.md](MODELS.md)** dosyasına bakınız.

### Notebook adlandırması hakkında

Notebooklar Drive'daki özgün dosyalardan **kopyalanmış** ve içeriklerine göre
yeniden adlandırılmıştır. Özgün dosya adları sürüm numaralarıyla uyuşmuyordu:

| Depodaki ad | Drive'daki özgün ad |
|---|---|
| `01_mimari_karsilastirmasi.ipynb` | `Tez_Model.ipynb` |
| `02_cozunurluk_ve_kutucuklama.ipynb` | `Tez_Modelv3.ipynb` |
| `03_nihai_model.ipynb` | `Tez_Model_v6.ipynb` |
| `04_analiz_ve_degerlendirme.ipynb` | `Untitled2.ipynb` |
| `05_alan_kaymasi_denemesi.ipynb` | `Tez_Modelv4.ipynb` |

---

## 4. Kurulum

```bash
git clone https://github.com/<kullanici>/tez-arac-hasar-tespit.git
cd tez-arac-hasar-tespit
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Model ağırlıklarını `models/` klasörüne indirin (bkz. MODELS.md):

```
models/
├── model1_parca_konum_yolo11m-seg.pt
└── model2_hasar_nihai_yolo11l-seg_fp32.pt
```

---

## 5. Kullanım

### Tek görüntü

```bash
python predict.py --image ornek.jpg
```

### Klasör, JSON çıktısı ve görselleştirme ile

```bash
python predict.py \
    --image ./test_goruntuleri/ \
    --out ./sonuclar/ \
    --json sonuclar.json \
    --conf 0.25
```

### Yalnızca hasar modeli (parça eşlemesi olmadan)

```bash
python predict.py --image ornek.jpg --no-parts
```

Örnek çıktı:

```
ornek.jpg
  [1] Gocuk  guven=0.87  parca=sol_on_kapi (ortusme=0.74)
  [2] Cizik  guven=0.61  parca=sol_on_camurluk (ortusme=0.52)
  2 hasar bulundu — sonuclar/ornek_sonuc.jpg
```

### Python'dan

```python
from predict import HasarTespitSistemi

sistem = HasarTespitSistemi(
    parca_model="models/model1_parca_konum_yolo11m-seg.pt",
    hasar_model="models/model2_hasar_nihai_yolo11l-seg_fp32.pt",
)
sonuc = sistem.tahmin("ornek.jpg")
for h in sonuc["hasarlar"]:
    print(h["sinif"], h["guven"], h["parca"])
```

---

## 6. Yeniden Üretim

Notebooklar Google Colab için yazılmıştır ve Drive bağlama adımı içerir.
Nihai modeli sıfırdan eğitmek için `notebooks/03_nihai_model.ipynb` yeterlidir;
hücreler sırayla çalıştırılmalıdır.

Veri seti Roboflow üzerinde barındırılmaktadır. Notebookların veri indirme
hücreleri bir Roboflow API anahtarı bekler; bu anahtar depoda **yer almaz** ve
ortam değişkeni olarak verilmelidir:

```bash
export ROBOFLOW_API_KEY="..."
```

Colab'da:

```python
import os
from google.colab import userdata
os.environ["ROBOFLOW_API_KEY"] = userdata.get("ROBOFLOW_API_KEY")
```

### Donanım

Eğitim Google Colab Pro+ üzerinde yapılmıştır. Nihai modelin toplam eğitim
süresi yaklaşık 60 saattir (150 devir fp16 + 70 devir fp32 devam eğitimi).

---

## 7. Bilinen Sınırlılıklar

Ayrıntılar için [MODELS.md](MODELS.md) → "Sınırlılıklar" bölümüne bakınız.
Özetle:

1. **Sürümler arası karşılaştırma** — her sürümde veri bölme hücresi yeniden
   çalıştırıldığı için (`seed=42` sabit olmasına rağmen havuz değiştiğinden)
   test kümeleri sürümler arasında birebir aynı değildir. Ablasyon tablosundaki
   karşılaştırmalar bu nedenle göstergesel kabul edilmelidir.
2. **Alan kayması** — farklı kaynaklardan gelen harici veri eklendiğinde
   başarım düşmektedir (bkz. `05_alan_kaymasi_denemesi.ipynb`).
3. **Çizik sınıfı** — en düşük başarımlı sınıftır (0.6381); ince çizikler düşük
   çözünürlükte kaybolmaktadır.
4. **fp16 sayısal kararsızlığı** — karışık hassasiyetli eğitimde maske kaybı
   taşması gözlenmiştir; nihai eğitim fp32 (`amp=False`) ile tamamlanmıştır.

---

## 8. Atıf

```bibtex
@mastersthesis{ozdemir2026hasar,
  author  = {Özdemir, Vesalettin Burhan},
  title   = {Sigorta Sektöründe Derin Öğrenme Tabanlı Otomatik
             Araç Hasar Tespit Sistemi},
  school  = {Üsküdar Üniversitesi, Fen Bilimleri Enstitüsü},
  address = {İstanbul, Türkiye},
  year    = {2026},
  type    = {Yüksek Lisans Tezi},
}
```

---

## 9. Lisans ve Teşekkür

Kod MIT lisansı ile paylaşılmıştır. Eğitilmiş model ağırlıkları yalnızca
akademik kullanım içindir.

Bu çalışma [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
kütüphanesi üzerine kurulmuştur.
