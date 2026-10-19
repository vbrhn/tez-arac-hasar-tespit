#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sigorta Sektöründe Derin Öğrenme Tabanlı Otomatik Araç Hasar Tespit Sistemi
Tek komutluk çıkarım scripti.

İki aşamalı füzyon:
    Model 1 (YOLOv11m-seg)  -> araç parçaları  (kaput, kapı, tampon, ...)
    Model 2 (YOLOv11l-seg)  -> hasarlar        (Çizik, Göçük, Kırık)
    Füzyon                  -> hangi hasar hangi parçada

Örtüşme oranı, hasar poligonunun parça poligonu ile kesişim alanının hasar
alanına bölümüdür. Hasar, en yüksek örtüşmeyi veren parçaya atanır; bu oran
eşiğin (varsayılan 0.10) altında kalırsa konum belirsiz olarak raporlanır.

Kullanım:
    python predict.py --image ornek.jpg
    python predict.py --image ./goruntuler/ --out ./sonuclar/ --json sonuc.json
    python predict.py --image ornek.jpg --no-parts        # yalnızca hasar modeli

Python'dan:
    from predict import HasarTespitSistemi
    sistem = HasarTespitSistemi()
    sonuc = sistem.tahmin("ornek.jpg")

Yazar : Vesalettin Burhan Özdemir
Kurum : Üsküdar Üniversitesi, Fen Bilimleri Enstitüsü
Yıl   : 2026
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

import numpy as np

# ----------------------------------------------------------------------------
# Varsayılanlar
# ----------------------------------------------------------------------------

VARSAYILAN_PARCA_MODEL = "models/model1_parca_konum_yolo11m-seg.pt"
VARSAYILAN_HASAR_MODEL = "models/model2_hasar_nihai_yolo11l-seg_fp32.pt"

VARSAYILAN_GUVEN = 0.25          # Ultralytics conf
VARSAYILAN_IOU = 0.45            # NMS IoU
VARSAYILAN_IMGSZ = 960           # eğitimdeki çözünürlük
VARSAYILAN_ORTUSME_ESIGI = 0.10  # tezde kullanılan operasyonel füzyon eşiği (Bölüm 3.3.3)

DESTEKLENEN_UZANTILAR = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Görselleştirme renkleri (BGR)
SINIF_RENKLERI = {
    "Cizik": (255, 170, 60),
    "Gocuk": (90, 200, 255),
    "Kirik": (90, 90, 245),
}
VARSAYILAN_RENK = (200, 200, 200)

# Model sınıf adları ASCII'dir (Cizik/Gocuk/Kirik); ekranda Türkçe gösterilir.
GORUNEN_AD = {"Cizik": "Çizik", "Gocuk": "Göçük", "Kirik": "Kırık"}
PARCA_RENGI = (170, 170, 170)


# ----------------------------------------------------------------------------
# Veri yapıları
# ----------------------------------------------------------------------------


@dataclass
class ParcaAtamasi:
    """Bir hasara atanan tek bir araç parçası."""

    parca: str
    ortusme: float
    parca_guveni: float


@dataclass
class Hasar:
    """Tespit edilen tek bir hasar örneği."""

    sinif: str
    guven: float
    alan_piksel: float
    kutu: list[float]  # [x1, y1, x2, y2]
    parca: str | None = None
    parca_ortusmesi: float | None = None
    tum_parcalar: list[ParcaAtamasi] = field(default_factory=list)


# ----------------------------------------------------------------------------
# Geometri yardımcıları
# ----------------------------------------------------------------------------


def _poligon_olustur(noktalar: np.ndarray):
    """Ultralytics maske poligonunu geçerli bir Shapely poligonuna çevirir.

    Kendisiyle kesişen poligonlar `buffer(0)` ile onarılır; onarılamayanlar
    için None döner.
    """
    from shapely.geometry import Polygon

    if noktalar is None or len(noktalar) < 3:
        return None

    try:
        p = Polygon(noktalar)
    except Exception:
        return None

    if not p.is_valid:
        p = p.buffer(0)

    if p.is_empty or p.area <= 0:
        return None

    # buffer(0) bazen MultiPolygon döndürür; en büyük parçayı al
    if p.geom_type == "MultiPolygon":
        p = max(p.geoms, key=lambda g: g.area)

    return p if p.area > 0 else None


def ortusme_orani(hasar_poligonu, parca_poligonu) -> float:
    """alan(hasar ∩ parça) / alan(hasar)

    Paydanın hasar alanı olması bilinçlidir: küçük bir hasarın büyük bir
    parçanın içinde tamamen kalması durumunda oran 1.0 olmalıdır.
    """
    if hasar_poligonu is None or parca_poligonu is None:
        return 0.0
    if hasar_poligonu.area <= 0:
        return 0.0
    try:
        kesisim = hasar_poligonu.intersection(parca_poligonu)
    except Exception:
        return 0.0
    if kesisim.is_empty:
        return 0.0
    return float(kesisim.area / hasar_poligonu.area)


# ----------------------------------------------------------------------------
# Ana sistem
# ----------------------------------------------------------------------------


class HasarTespitSistemi:
    """İki aşamalı araç hasar tespit sistemi."""

    def __init__(
        self,
        hasar_model: str | Path = VARSAYILAN_HASAR_MODEL,
        parca_model: str | Path | None = VARSAYILAN_PARCA_MODEL,
        *,
        guven: float = VARSAYILAN_GUVEN,
        iou: float = VARSAYILAN_IOU,
        imgsz: int = VARSAYILAN_IMGSZ,
        ortusme_esigi: float = VARSAYILAN_ORTUSME_ESIGI,
        cihaz: str | None = None,
        sessiz: bool = True,
    ) -> None:
        from ultralytics import YOLO

        hasar_yolu = Path(hasar_model)
        if not hasar_yolu.exists():
            raise FileNotFoundError(
                f"Hasar modeli bulunamadı: {hasar_yolu}\n"
                "Ağırlıkları indirip models/ klasörüne koyun (bkz. MODELS.md)."
            )

        self.guven = guven
        self.iou = iou
        self.imgsz = imgsz
        self.ortusme_esigi = ortusme_esigi
        self.cihaz = cihaz
        self.sessiz = sessiz

        self.hasar_modeli = YOLO(str(hasar_yolu))
        self.parca_modeli = None

        if parca_model is not None:
            parca_yolu = Path(parca_model)
            if parca_yolu.exists():
                self.parca_modeli = YOLO(str(parca_yolu))
            else:
                print(
                    f"[uyarı] Parça modeli bulunamadı: {parca_yolu}\n"
                    "         Hasarlar parça eşlemesi olmadan raporlanacak.",
                    file=sys.stderr,
                )

    # -- iç yardımcılar ----------------------------------------------------

    def _calistir(self, model, goruntu):
        kwargs: dict[str, Any] = dict(
            conf=self.guven,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=not self.sessiz,
            retina_masks=True,
        )
        if self.cihaz:
            kwargs["device"] = self.cihaz
        return model.predict(goruntu, **kwargs)[0]

    @staticmethod
    def _ornekleri_cikar(sonuc) -> list[dict[str, Any]]:
        """Ultralytics sonucundan (sınıf, güven, poligon, kutu) listesi üretir."""
        ornekler: list[dict[str, Any]] = []
        if sonuc.masks is None or sonuc.boxes is None:
            return ornekler

        adlar = sonuc.names
        poligonlar = sonuc.masks.xy
        kutular = sonuc.boxes

        for i, nokta_dizisi in enumerate(poligonlar):
            poligon = _poligon_olustur(np.asarray(nokta_dizisi, dtype=float))
            if poligon is None:
                continue
            sinif_id = int(kutular.cls[i].item())
            ornekler.append(
                {
                    "sinif": adlar.get(sinif_id, str(sinif_id)),
                    "guven": float(kutular.conf[i].item()),
                    "poligon": poligon,
                    "kutu": [round(float(v), 1) for v in kutular.xyxy[i].tolist()],
                    "noktalar": np.asarray(nokta_dizisi, dtype=np.int32),
                }
            )
        return ornekler

    # -- genel API ---------------------------------------------------------

    def tahmin(self, goruntu_yolu: str | Path) -> dict[str, Any]:
        """Tek bir görüntüyü işler ve sonuç sözlüğü döndürür."""
        goruntu_yolu = Path(goruntu_yolu)
        if not goruntu_yolu.exists():
            raise FileNotFoundError(f"Görüntü bulunamadı: {goruntu_yolu}")

        hasar_sonucu = self._calistir(self.hasar_modeli, str(goruntu_yolu))
        hasar_ornekleri = self._ornekleri_cikar(hasar_sonucu)

        parca_ornekleri: list[dict[str, Any]] = []
        if self.parca_modeli is not None:
            parca_sonucu = self._calistir(self.parca_modeli, str(goruntu_yolu))
            parca_ornekleri = self._ornekleri_cikar(parca_sonucu)

        eslesmis: list[tuple[Hasar, dict[str, Any]]] = []
        for h in hasar_ornekleri:
            atamalar: list[ParcaAtamasi] = []
            for p in parca_ornekleri:
                oran = ortusme_orani(h["poligon"], p["poligon"])
                if oran >= self.ortusme_esigi:
                    atamalar.append(
                        ParcaAtamasi(
                            parca=p["sinif"],
                            ortusme=round(oran, 4),
                            parca_guveni=round(p["guven"], 4),
                        )
                    )

            atamalar.sort(key=lambda a: a.ortusme, reverse=True)

            eslesmis.append(
                (
                    Hasar(
                        sinif=h["sinif"],
                        guven=round(h["guven"], 4),
                        alan_piksel=round(h["poligon"].area, 1),
                        kutu=h["kutu"],
                        parca=atamalar[0].parca if atamalar else None,
                        parca_ortusmesi=atamalar[0].ortusme if atamalar else None,
                        tum_parcalar=atamalar,
                    ),
                    h,
                )
            )

        # Hasar kayıtları ile çizim geometrileri aynı sırayı korumalıdır.
        eslesmis.sort(key=lambda ikili: ikili[0].guven, reverse=True)
        hasarlar = [ikili[0] for ikili in eslesmis]
        cizim_geometrileri = [ikili[1] for ikili in eslesmis]

        yukseklik, genislik = hasar_sonucu.orig_shape

        return {
            "goruntu": goruntu_yolu.name,
            "yol": str(goruntu_yolu),
            "boyut": {"genislik": int(genislik), "yukseklik": int(yukseklik)},
            "hasar_sayisi": len(hasarlar),
            "hasarlar": [asdict(d) for d in hasarlar],
            "tespit_edilen_parcalar": sorted({p["sinif"] for p in parca_ornekleri}),
            "ayarlar": {
                "guven": self.guven,
                "iou": self.iou,
                "imgsz": self.imgsz,
                "ortusme_esigi": self.ortusme_esigi,
            },
            "_ic": {"hasar": cizim_geometrileri, "parca": parca_ornekleri},
        }

    def gorsellestir(
        self,
        goruntu_yolu: str | Path,
        sonuc: dict[str, Any],
        cikti_yolu: str | Path,
        *,
        parcalari_ciz: bool = False,
    ) -> Path:
        """Hasar maskelerini görüntü üzerine çizer ve diske yazar."""
        import cv2

        goruntu = cv2.imread(str(goruntu_yolu))
        if goruntu is None:
            raise ValueError(f"Görüntü okunamadı: {goruntu_yolu}")

        katman = goruntu.copy()

        if parcalari_ciz:
            for p in sonuc["_ic"]["parca"]:
                cv2.polylines(katman, [p["noktalar"]], True, PARCA_RENGI, 2)

        for h_ic, h in zip(sonuc["_ic"]["hasar"], sonuc["hasarlar"], strict=False):
            renk = SINIF_RENKLERI.get(h_ic["sinif"], VARSAYILAN_RENK)
            cv2.fillPoly(katman, [h_ic["noktalar"]], renk)
            cv2.polylines(goruntu, [h_ic["noktalar"]], True, renk, 2)

        goruntu = cv2.addWeighted(katman, 0.4, goruntu, 0.6, 0)

        for h_ic, h in zip(sonuc["_ic"]["hasar"], sonuc["hasarlar"], strict=False):
            renk = SINIF_RENKLERI.get(h_ic["sinif"], VARSAYILAN_RENK)
            x1, y1 = int(h_ic["kutu"][0]), int(h_ic["kutu"][1])
            etiket = f"{h_ic['sinif']} {h_ic['guven']:.2f}"
            if h.get("parca"):
                etiket += f" | {h['parca']}"
            (tw, th), _ = cv2.getTextSize(etiket, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(goruntu, (x1, y1 - th - 8), (x1 + tw + 6, y1), renk, -1)
            cv2.putText(
                goruntu,
                etiket,
                (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (20, 20, 20),
                1,
                cv2.LINE_AA,
            )

        cikti_yolu = Path(cikti_yolu)
        cikti_yolu.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(cikti_yolu), goruntu)
        return cikti_yolu


# ----------------------------------------------------------------------------
# Komut satırı
# ----------------------------------------------------------------------------


def _goruntuleri_topla(hedef: Path) -> list[Path]:
    if hedef.is_file():
        return [hedef]
    if hedef.is_dir():
        return sorted(
            p for p in hedef.rglob("*") if p.suffix.lower() in DESTEKLENEN_UZANTILAR
        )
    raise FileNotFoundError(f"Bulunamadı: {hedef}")


def _temizle(sonuc: dict[str, Any]) -> dict[str, Any]:
    """JSON'a yazmadan önce iç geometri nesnelerini atar."""
    return {k: v for k, v in sonuc.items() if k != "_ic"}


def main(argv: list[str] | None = None) -> int:
    ayristirici = argparse.ArgumentParser(
        description="Otomatik araç hasar tespit sistemi — iki aşamalı YOLOv11-seg",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Örnekler:\n"
            "  python predict.py --image ornek.jpg\n"
            "  python predict.py --image ./goruntuler/ --out ./sonuclar/\n"
            "  python predict.py --image ornek.jpg --json sonuc.json --no-parts\n"
        ),
    )
    ayristirici.add_argument(
        "--image", "-i", required=True, help="görüntü dosyası veya klasörü"
    )
    ayristirici.add_argument(
        "--out", "-o", default=None, help="görselleştirmelerin yazılacağı klasör"
    )
    ayristirici.add_argument("--json", "-j", default=None, help="JSON çıktı dosyası")
    ayristirici.add_argument("--hasar-model", default=VARSAYILAN_HASAR_MODEL)
    ayristirici.add_argument("--parca-model", default=VARSAYILAN_PARCA_MODEL)
    ayristirici.add_argument(
        "--no-parts",
        action="store_true",
        help="parça modelini kullanma, yalnızca hasar tespiti yap",
    )
    ayristirici.add_argument("--conf", type=float, default=VARSAYILAN_GUVEN)
    ayristirici.add_argument("--iou", type=float, default=VARSAYILAN_IOU)
    ayristirici.add_argument("--imgsz", type=int, default=VARSAYILAN_IMGSZ)
    ayristirici.add_argument(
        "--overlap",
        type=float,
        default=VARSAYILAN_ORTUSME_ESIGI,
        help=f"füzyon örtüşme eşiği (varsayılan {VARSAYILAN_ORTUSME_ESIGI})",
    )
    ayristirici.add_argument(
        "--device", default=None, help="cuda, cuda:0, cpu, mps (varsayılan: otomatik)"
    )
    ayristirici.add_argument(
        "--draw-parts", action="store_true", help="parça sınırlarını da çiz"
    )
    args = ayristirici.parse_args(argv)

    try:
        goruntuler = _goruntuleri_topla(Path(args.image))
    except FileNotFoundError as hata:
        print(f"hata: {hata}", file=sys.stderr)
        return 1

    if not goruntuler:
        print("hata: işlenecek görüntü bulunamadı.", file=sys.stderr)
        return 1

    try:
        sistem = HasarTespitSistemi(
            hasar_model=args.hasar_model,
            parca_model=None if args.no_parts else args.parca_model,
            guven=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            ortusme_esigi=args.overlap,
            cihaz=args.device,
        )
    except FileNotFoundError as hata:
        print(f"hata: {hata}", file=sys.stderr)
        return 1

    cikti_dizini = Path(args.out) if args.out else None
    if cikti_dizini:
        cikti_dizini.mkdir(parents=True, exist_ok=True)

    tum_sonuclar: list[dict[str, Any]] = []
    toplam_hasar = 0

    for yol in goruntuler:
        try:
            sonuc = sistem.tahmin(yol)
        except Exception as hata:  # tek bir bozuk dosya tüm koşuyu düşürmesin
            print(f"{yol.name}: atlandı ({hata})", file=sys.stderr)
            continue

        toplam_hasar += sonuc["hasar_sayisi"]
        print(f"\n{yol.name}")

        if sonuc["hasar_sayisi"] == 0:
            print("  hasar bulunamadı")
        else:
            for sira, h in enumerate(sonuc["hasarlar"], 1):
                satir = f"  [{sira}] {GORUNEN_AD.get(h['sinif'], h['sinif']):<6} güven={h['guven']:.2f}"
                if h["parca"]:
                    satir += f"  parça={h['parca']} (örtüşme={h['parca_ortusmesi']:.2f})"
                elif sistem.parca_modeli is not None:
                    satir += "  parça=belirsiz"
                print(satir)

        if cikti_dizini:
            hedef = cikti_dizini / f"{yol.stem}_sonuc.jpg"
            try:
                sistem.gorsellestir(yol, sonuc, hedef, parcalari_ciz=args.draw_parts)
                print(f"  -> {hedef}")
            except Exception as hata:
                print(f"  görselleştirme başarısız: {hata}", file=sys.stderr)

        tum_sonuclar.append(_temizle(sonuc))

    print(
        f"\n{len(tum_sonuclar)} görüntü işlendi, toplam {toplam_hasar} hasar tespit edildi."
    )

    if args.json:
        json_yolu = Path(args.json)
        json_yolu.parent.mkdir(parents=True, exist_ok=True)
        json_yolu.write_text(
            json.dumps(tum_sonuclar, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"JSON yazıldı: {json_yolu}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
