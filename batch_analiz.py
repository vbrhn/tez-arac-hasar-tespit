# -*- coding: utf-8 -*-
"""
Kullanicinin verdigi 13 fotograf icin:
  1. Iki asamali cikarim (hasar + parca) ve fuzyon
  2. Plaka bulaniklastirma (GDPR — herkese acik depoya gidiyor)
  3. Maskelenmis gorseller: TR ve EN etiketli
  4. Sonuclari JSON'a yaz (rapor uretici bunu kullanacak)
"""
import json, pathlib, warnings, pickle
warnings.filterwarnings('ignore')
import numpy as np, cv2
from shapely.geometry import Polygon
from ultralytics import YOLO

KAYNAK = pathlib.Path('/home/claude/kullanici_foto')
CIK    = pathlib.Path('/mnt/user-data/outputs/kullanici_sonuc')
(CIK / 'maskeli_TR').mkdir(parents=True, exist_ok=True)
(CIK / 'maskeli_EN').mkdir(parents=True, exist_ok=True)

UP = pathlib.Path('/mnt/user-data/uploads')
HASAR = YOLO(str(UP / 'Downloads/model2_hasar_nihai_yolo11l-seg_fp32.pt'))
PARCA = YOLO(str(UP / 'Downloads/model1_parca_konum_yolo11m-seg.pt'))

CONF, IOU, IMGSZ, ESIK = 0.25, 0.45, 960, 0.38

SINIF = {'TR': {'Cizik': 'Çizik', 'Gocuk': 'Göçük', 'Kirik': 'Kırık'},
         'EN': {'Cizik': 'Scratch', 'Gocuk': 'Dent', 'Kirik': 'Crack'}}
RENK  = {'Cizik': '#E8A33D', 'Gocuk': '#2EA8C7', 'Kirik': '#D1495B'}

PARCA_TR = {
    'back_bumper':'Arka Tampon','back_door':'Arka Kapı','back_glass':'Arka Cam',
    'back_left_door':'Sol Arka Kapı','back_left_light':'Sol Arka Far',
    'back_light':'Arka Far','back_right_door':'Sağ Arka Kapı',
    'back_right_light':'Sağ Arka Far','front_bumper':'Ön Tampon',
    'front_door':'Ön Kapı','front_glass':'Ön Cam','front_left_door':'Sol Ön Kapı',
    'front_left_light':'Sol Ön Far','front_light':'Ön Far',
    'front_right_door':'Sağ Ön Kapı','front_right_light':'Sağ Ön Far',
    'hood':'Kaput','left_mirror':'Sol Ayna','object':'Nesne',
    'right_mirror':'Sağ Ayna','tailgate':'Bagaj Kapağı','trunk':'Bagaj',
    'wheel':'Tekerlek'}

def parca_ad(h, dil):
    return PARCA_TR.get(h, h.replace('_', ' ').title()) if dil == 'TR' \
           else h.replace('_', ' ').title()

# ── ARAÇLAR ────────────────────────────────────────────────────────────────
ARACLAR = [
 dict(kod='242866', tr='Araç 1 — Audi RS6 Avant (siyah)', en='Vehicle 1 — Audi RS6 Avant (black)',
      marka='Audi', model='RS6 Avant', renk_tr='Siyah', renk_en='Black',
      foto=['242866_1.jpg', '242866_2.jpg', '242866_4.jpg'],
      aci_tr=['Arka sağ çeyrek', 'Ön sağ çeyrek', 'Ön'],
      aci_en=['Rear right quarter', 'Front right quarter', 'Front']),
 dict(kod='17606914461334369139', tr='Araç 2 — Peugeot 108 (gümüş)', en='Vehicle 2 — Peugeot 108 (silver)',
      marka='Peugeot', model='108', renk_tr='Gümüş', renk_en='Silver',
      foto=['17606914461334369139-2.jpg', '17606914461334369139-3.jpg',
            '17606914461334369139-5.jpg'],
      aci_tr=['Ön', 'Ön sağ çeyrek', 'Arka sağ çeyrek'],
      aci_en=['Front', 'Front right quarter', 'Rear right quarter']),
 dict(kod='1760712142464517150', tr='Araç 3 — Peugeot Partner Tepee (mavi)', en='Vehicle 3 — Peugeot Partner Tepee (blue)',
      marka='Peugeot', model='Partner Tepee', renk_tr='Mavi', renk_en='Blue',
      foto=['1760712142464517150-1.jpg', '1760712142464517150-3.jpg',
            '1760712142464517150-5.jpg', '1760712142464517150-8.jpg'],
      aci_tr=['Arka sol çeyrek', 'Ön sol çeyrek', 'Ön sağ çeyrek', 'Arka sağ çeyrek'],
      aci_en=['Rear left quarter', 'Front left quarter', 'Front right quarter',
              'Rear right quarter']),
 dict(kod='1761133610501428962', tr='Araç 4 — Hyundai Tucson (bej)', en='Vehicle 4 — Hyundai Tucson (beige)',
      marka='Hyundai', model='Tucson', renk_tr='Bej', renk_en='Beige',
      foto=['1761133610501428962-2.jpg', '1761133610501428962-5.jpg',
            '1761133610501428962-11.jpg'],
      aci_tr=['Arka', 'Ön', 'Ön sağ çeyrek'],
      aci_en=['Rear', 'Front', 'Front right quarter']),
]

# ── PLAKA BÖLGELERİ (normalize x1,y1,x2,y2) — GDPR için bulanıklaştırılır ──
PLAKA = {
 '17606914461334369139-2.jpg': [(0.345, 0.750, 0.660, 0.870)],
 '17606914461334369139-3.jpg': [(0.730, 0.645, 0.900, 0.800)],
 '1761133610501428962-2.jpg':  [(0.350, 0.590, 0.655, 0.690)],
 '1761133610501428962-5.jpg':  [(0.325, 0.880, 0.710, 0.980)],
 '1761133610501428962-11.jpg': [(0.740, 0.690, 0.930, 0.795)],
}

def plaka_gizle(im, ad):
    for (a, b, c, d) in PLAKA.get(ad, []):
        H, W = im.shape[:2]
        x1, y1, x2, y2 = int(a*W), int(b*H), int(c*W), int(d*H)
        roi = im[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        k = max(3, (x2-x1)//12)
        kucuk = cv2.resize(roi, (max(1,(x2-x1)//k), max(1,(y2-y1)//k)),
                           interpolation=cv2.INTER_LINEAR)
        im[y1:y2, x1:x2] = cv2.resize(kucuk, (x2-x1, y2-y1),
                                      interpolation=cv2.INTER_NEAREST)
    return im

# ── Geometri ───────────────────────────────────────────────────────────────
def poly(xy):
    if xy is None or len(xy) < 3:
        return None
    try:
        p = Polygon(np.asarray(xy, float))
    except Exception:
        return None
    if not p.is_valid:
        p = p.buffer(0)
    if p.is_empty or p.area <= 0:
        return None
    if p.geom_type == 'MultiPolygon':
        p = max(p.geoms, key=lambda g: g.area)
    return p if p.area > 0 else None

def hex2bgr(h):
    h = h.lstrip('#')
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)

def ciz(im_bgr, hasarlar, parcalar, dil):
    out = im_bgr.copy()
    kat = im_bgr.copy()
    for p in parcalar:
        cv2.polylines(kat, [p['pts']], True, (160, 160, 160), 2)
    for h in hasarlar:
        c = hex2bgr(RENK.get(h['ham'], '#888888'))
        cv2.fillPoly(kat, [h['pts']], c)
        cv2.polylines(out, [h['pts']], True, c, 3)
    out = cv2.addWeighted(kat, 0.42, out, 0.58, 0)

    H, W = im_bgr.shape[:2]
    ol = max(0.55, min(W, H) / 1050)
    kal = max(1, int(ol * 2))
    for h in hasarlar:
        c = hex2bgr(RENK.get(h['ham'], '#888888'))
        met = f"{SINIF[dil].get(h['ham'], h['ham'])} {h['guven']:.2f}"
        if h.get('parca'):
            met += f" | {parca_ad(h['parca'], dil)}"
        (tw, th), _ = cv2.getTextSize(met, cv2.FONT_HERSHEY_DUPLEX, ol, kal)
        x1 = max(4, min(int(h['kutu'][0]), W - tw - 14))
        y1 = min(max(int(h['kutu'][1]), th + 12), H - 6)
        cv2.rectangle(out, (x1, y1 - th - 10), (x1 + tw + 12, y1), c, -1)
        cv2.putText(out, met, (x1 + 6, y1 - 7), cv2.FONT_HERSHEY_DUPLEX,
                    ol, (255, 255, 255), kal, cv2.LINE_AA)
    return out

def ornekleri_al(res):
    out = []
    if res.masks is None or res.boxes is None:
        return out
    for i, xy in enumerate(res.masks.xy):
        pol = poly(xy)
        if pol is None:
            continue
        out.append(dict(ham=res.names[int(res.boxes.cls[i].item())],
                        guven=float(res.boxes.conf[i].item()),
                        pol=pol, pts=np.asarray(xy, np.int32),
                        kutu=[float(v) for v in res.boxes.xyxy[i].tolist()],
                        idx=i))
    return out

# ════════════════════════════════════════════════════════════════════════════
tum = []
print('ANALİZ\n' + '=' * 78)

for arac in ARACLAR:
    print(f"\n{arac['tr']}")
    kayit_foto = []
    for j, ad in enumerate(arac['foto']):
        yol = KAYNAK / ad
        im = cv2.imread(str(yol))
        im = plaka_gizle(im, ad)
        H, W = im.shape[:2]

        rh = HASAR.predict(im, conf=CONF, iou=IOU, imgsz=IMGSZ,
                           verbose=False, retina_masks=True)[0]
        rp = PARCA.predict(im, conf=CONF, iou=IOU, imgsz=IMGSZ,
                           verbose=False, retina_masks=True)[0]
        hasarlar = ornekleri_al(rh)
        parcalar = ornekleri_al(rp)

        for h in hasarlar:
            en_iyi, oran_iyi = None, 0.0
            for p in parcalar:
                k = h['pol'].intersection(p['pol'])
                o = 0.0 if k.is_empty else k.area / h['pol'].area
                if o > oran_iyi:
                    en_iyi, oran_iyi = p, o
            h['parca']   = en_iyi['ham'] if (en_iyi and oran_iyi >= ESIK) else None
            h['ortusme'] = oran_iyi if h['parca'] else None
            m = rh.masks.data[h['idx']].cpu().numpy()
            h['alan_yuzde'] = 100.0 * float((cv2.resize(m, (W, H)) > .5).sum()) / (W * H)
        hasarlar.sort(key=lambda h: h['guven'], reverse=True)

        for dil in ('TR', 'EN'):
            cv2.imwrite(str(CIK / f'maskeli_{dil}' / f'{pathlib.Path(ad).stem}_{dil}.jpg'),
                        ciz(im, hasarlar, parcalar, dil),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])

        cv2.imwrite(str(CIK / f'_tmp_{pathlib.Path(ad).stem}.jpg'),
                    ciz(im, hasarlar, parcalar, 'TR'), [cv2.IMWRITE_JPEG_QUALITY, 90])

        print(f"   [{j+1}] {ad:<30} {len(hasarlar)} hasar, {len(parcalar)} parça")
        for h in hasarlar:
            pr = f"{h['parca']} ({h['ortusme']:.2f})" if h['parca'] else '—'
            print(f"        {h['ham']:<6} {h['guven']:.2f}  alan={h['alan_yuzde']:.2f}%  parça={pr}")

        kayit_foto.append(dict(
            dosya=ad, aci_tr=arac['aci_tr'][j], aci_en=arac['aci_en'][j],
            boyut=[W, H],
            hasarlar=[dict(ham=h['ham'], guven=round(h['guven'], 4),
                           parca=h['parca'],
                           ortusme=round(h['ortusme'], 4) if h['ortusme'] else None,
                           alan_yuzde=round(h['alan_yuzde'], 3)) for h in hasarlar],
            parcalar=sorted({p['ham'] for p in parcalar})))
    tum.append(dict(**{k: v for k, v in arac.items() if k != 'foto'},
                    foto=arac['foto'], sonuc=kayit_foto))

(CIK / 'analiz_sonuc.json').write_text(
    json.dumps(tum, ensure_ascii=False, indent=1), encoding='utf-8')

print('\n' + '=' * 78)
n_h = sum(len(f['hasarlar']) for a in tum for f in a['sonuc'])
print(f'{len(ARACLAR)} araç, {sum(len(a["foto"]) for a in ARACLAR)} fotoğraf, '
      f'{n_h} hasar örneği')
print(f'Plaka bulanıklaştırılan: {len(PLAKA)} fotoğraf')
print(f'Çıktı: {CIK}')
