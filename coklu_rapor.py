# -*- coding: utf-8 -*-
"""
ÇOK ARAÇLI HASAR RAPORU  /  MULTI-VEHICLE DAMAGE REPORT
=========================================================
Tek bir TRAMER formatlı PDF içinde birden fazla araç ve her araç için birden
fazla fotoğraf. DIL='TR' | 'EN'.

Kullanım:
    python coklu_rapor.py            # her iki dilde üretir
"""
import json, os, sys, pathlib
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image as RLImage, PageBreak,
                                KeepTogether)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

SONUC = pathlib.Path(os.environ.get('CIKTI_DIZINI', 'kullanici_sonuc'))
CIK   = SONUC

SINIF = {'TR': {'Cizik': 'Çizik', 'Gocuk': 'Göçük', 'Kirik': 'Kırık'},
         'EN': {'Cizik': 'Scratch', 'Gocuk': 'Dent', 'Kirik': 'Crack'}}

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
    return PARCA_TR.get(h, h.replace('_',' ').title()) if dil=='TR' \
           else h.replace('_',' ').title()

M = {
'TR': dict(
  baslik='ÇOKLU ARAÇ HASAR TESPİT RAPORU',
  alt='Yapay Zekâ Destekli Otomatik Ekspertiz Sistemi — Toplu Değerlendirme',
  belge='Belge No', tarih='Düzenleme Tarihi', tur='Rapor Türü',
  tur_v='Otomatik Ön Tespit (toplu)', kapsam='Kapsam',
  kapsam_v='{a} araç / {f} fotoğraf',
  s_ozet='GENEL ÖZET', o_arac='Araç', o_marka='Marka / Model', o_foto='Fotoğraf',
  o_hasar='Hasar', o_tip='Tespit Edilen Tipler', o_alan='Topl. Alan (%)',
  s_arac='ARAÇ {n} — {ad}', b_marka='Marka', b_model='Model', b_renk='Renk',
  b_foto='Fotoğraf Sayısı', b_hasar='Toplam Hasar',
  t_foto='Foto', t_aci='Çekim Açısı', t_parca='Parça', t_hasar='Hasar Tipi',
  t_alan='Alan (%)', t_guven='Güven (%)', t_durum='Durum',
  kesin='Kesin', olasi='Olası', supheli='Şüpheli', atanmadi='atanmadı',
  yok='Bu açıdan hasar tespit edilmedi.',
  g_baslik='Görsel Analiz', renk_k='Renk kodları: Çizik | Göçük | Kırık',
  s_sistem='SİSTEM BİLGİSİ',
  m_parca='Parça Tespit Modeli', m_hasar='Hasar Tespit Modeli',
  m_map='Hasar Modeli mAP@50 (test)', m_sure='Çıkarım Süresi',
  m_esik='Füzyon Örtüşme Eşiği', m_coz='Çıkarım Çözünürlüğü',
  s_not='AÇIKLAMALAR',
  yasal='<b>ÖNEMLİ AÇIKLAMA:</b> Bu rapor yapay zekâ tabanlı görüntü analiz sistemi '
        'tarafından otomatik olarak üretilmiştir ve <b>ön değerlendirme</b> '
        'niteliğindedir. Nihai hasar tespiti, onarım kapsamı ve tazminat tutarı '
        'yetkili eksper incelemesi sonucunda belirlenir. Sistem çıktıları yalnızca '
        'eksper iş akışını hızlandırmak amacıyla referans olarak kullanılmalıdır. '
        'Görüntü kalitesi, çekim açısı ve ışık koşulları tespit doğruluğunu '
        'etkileyebilir.',
  dipnot='Sayfa {s}  ·  {b}'),
'EN': dict(
  baslik='MULTI-VEHICLE DAMAGE ASSESSMENT REPORT',
  alt='AI-Assisted Automatic Insurance Appraisal System — Batch Assessment',
  belge='Document No', tarih='Issue Date', tur='Report Type',
  tur_v='Automated Preliminary Assessment (batch)', kapsam='Scope',
  kapsam_v='{a} vehicles / {f} photographs',
  s_ozet='OVERALL SUMMARY', o_arac='Vehicle', o_marka='Make / Model', o_foto='Photos',
  o_hasar='Damage', o_tip='Detected Types', o_alan='Total Area (%)',
  s_arac='VEHICLE {n} — {ad}', b_marka='Make', b_model='Model', b_renk='Colour',
  b_foto='Photograph Count', b_hasar='Total Damage',
  t_foto='Photo', t_aci='View Angle', t_parca='Part', t_hasar='Damage Type',
  t_alan='Area (%)', t_guven='Confidence (%)', t_durum='Status',
  kesin='Confirmed', olasi='Probable', supheli='Uncertain', atanmadi='unassigned',
  yok='No damage detected from this angle.',
  g_baslik='Visual Analysis', renk_k='Colour key: Scratch | Dent | Crack',
  s_sistem='SYSTEM INFORMATION',
  m_parca='Part Detection Model', m_hasar='Damage Detection Model',
  m_map='Damage Model mAP@50 (test)', m_sure='Inference Time',
  m_esik='Fusion Overlap Threshold', m_coz='Inference Resolution',
  s_not='NOTES',
  yasal='<b>IMPORTANT NOTICE:</b> This report was generated automatically by an '
        'AI-based image analysis system and constitutes a <b>preliminary assessment</b>. '
        'The final damage assessment, repair scope and settlement amount are determined '
        'by a qualified claims adjuster. System outputs should be used for reference '
        'only, to accelerate the adjuster workflow. Image quality, viewing angle and '
        'lighting conditions may affect detection accuracy.',
  dipnot='Page {s}  ·  {b}'),
}

SISTEM = dict(parca='YOLOv11m-seg (Carparts-Seg)',
              hasar='YOLOv11l-seg (v6, fp32)',
              map50='0.7238', sure='8.84 ms', esik='0.10', coz='960 px')

LACI, GRI, CIZGI = colors.HexColor('#1e40af'), colors.HexColor('#f8fafc'), colors.HexColor('#cbd5e1')


def font_hazirla(d='/home/claude/fonts'):
    os.makedirs(d, exist_ok=True)
    reg, bold = f'{d}/DejaVuSans.ttf', f'{d}/DejaVuSans-Bold.ttf'
    if not os.path.exists(reg):
        import shutil, urllib.request
        s = '/usr/share/fonts/truetype/dejavu'
        if os.path.exists(f'{s}/DejaVuSans.ttf'):
            shutil.copy(f'{s}/DejaVuSans.ttf', reg); shutil.copy(f'{s}/DejaVuSans-Bold.ttf', bold)
        else:
            b = 'https://raw.githubusercontent.com/dejavu-fonts/dejavu-fonts/master/build/ttf'
            urllib.request.urlretrieve(f'{b}/DejaVuSans.ttf', reg)
            urllib.request.urlretrieve(f'{b}/DejaVuSans-Bold.ttf', bold)
    pdfmetrics.registerFont(TTFont('DejaVu', reg))
    pdfmetrics.registerFont(TTFont('DejaVu-Bold', bold))



def _rapor_boyutu(yol, en=1100):
    """Rapora gomulecek gorseli kucult — PDF boyutu 3-4 kat duser,
    82 mm basim genisliginde hala ~340 dpi."""
    import cv2
    yol = pathlib.Path(yol)
    hedef = yol.parent.parent / '_rapor_kucuk' / yol.name
    hedef.parent.mkdir(parents=True, exist_ok=True)
    if not hedef.exists():
        im = cv2.imread(str(yol))
        h, w = im.shape[:2]
        if w > en:
            im = cv2.resize(im, (en, int(h * en / w)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(hedef), im, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return hedef


def coklu_rapor(veri, dil, dosya, belge_no=None):
    T = M[dil]
    simdi = datetime.now()
    belge_no = belge_no or f"{'HT' if dil=='TR' else 'VD'}-{simdi:%Y%m%d}-MULTI"

    ss = getSampleStyleSheet()
    S = dict(
        b=ParagraphStyle('b', parent=ss['Title'], fontName='DejaVu-Bold', fontSize=17,
                         textColor=colors.HexColor('#0f172a'), alignment=1, spaceAfter=2),
        a=ParagraphStyle('a', parent=ss['Normal'], fontName='DejaVu', fontSize=9.5,
                         textColor=colors.HexColor('#64748b'), alignment=1, spaceAfter=12),
        h=ParagraphStyle('h', parent=ss['Heading2'], fontName='DejaVu-Bold', fontSize=12,
                         textColor=LACI, spaceBefore=12, spaceAfter=6),
        h2=ParagraphStyle('h2', parent=ss['Heading3'], fontName='DejaVu-Bold', fontSize=9.5,
                          textColor=colors.HexColor('#334155'), spaceBefore=8, spaceAfter=4),
        p=ParagraphStyle('p', parent=ss['Normal'], fontName='DejaVu', fontSize=9, leading=13),
        k=ParagraphStyle('k', parent=ss['Normal'], fontName='DejaVu', fontSize=7.5,
                         textColor=colors.HexColor('#64748b'), leading=11),
    )

    def dipnot(canvas, doc):
        canvas.saveState()
        canvas.setFont('DejaVu', 7)
        canvas.setFillColor(colors.HexColor('#94a3b8'))
        canvas.drawCentredString(A4[0]/2, 10*mm,
                                 T['dipnot'].format(s=doc.page, b=belge_no))
        canvas.restoreState()

    doc = SimpleDocTemplate(dosya, pagesize=A4, leftMargin=16*mm, rightMargin=16*mm,
                            topMargin=14*mm, bottomMargin=16*mm,
                            title=T['baslik'], author='Vesalettin Burhan Ozdemir')
    st = [Paragraph(T['baslik'], S['b']), Paragraph(T['alt'], S['a'])]

    n_foto = sum(len(a['sonuc']) for a in veri)
    hs_ = ParagraphStyle('hs', parent=ss['Normal'], fontName='DejaVu',
                         fontSize=8.5, leading=10.5)
    P_ = lambda x: Paragraph(str(x), hs_)
    t0 = Table([[T['belge'], P_(belge_no), T['tarih'], P_(simdi.strftime('%d.%m.%Y %H:%M'))],
                [T['tur'], P_(T['tur_v']), T['kapsam'],
                 P_(T['kapsam_v'].format(a=len(veri), f=n_foto))]],
               colWidths=[28*mm, 58*mm, 26*mm, 56*mm])
    t0.setStyle(TableStyle([
        ('FONTNAME',(0,0),(-1,-1),'DejaVu'), ('FONTNAME',(0,0),(0,-1),'DejaVu-Bold'),
        ('FONTNAME',(2,0),(2,-1),'DejaVu-Bold'), ('FONTSIZE',(0,0),(-1,-1),8.5),
        ('BACKGROUND',(0,0),(0,-1),colors.HexColor('#f1f5f9')),
        ('BACKGROUND',(2,0),(2,-1),colors.HexColor('#f1f5f9')),
        ('GRID',(0,0),(-1,-1),0.4,CIZGI), ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),5), ('BOTTOMPADDING',(0,0),(-1,-1),5)]))
    st.append(t0)

    # ── GENEL ÖZET ──────────────────────────────────────────────────────────
    st.append(Paragraph(T['s_ozet'], S['h']))
    bs_ = ParagraphStyle('bs', parent=ss['Normal'], fontName='DejaVu-Bold',
                         fontSize=8.5, leading=10, textColor=colors.white)
    B_ = lambda x: Paragraph(str(x), bs_)
    satir = [[B_(T['o_arac']), B_(T['o_marka']), B_(T['o_foto']), B_(T['o_hasar']),
              B_(T['o_tip']), B_(T['o_alan'])]]
    for i, a in enumerate(veri, 1):
        hs = [h for f in a['sonuc'] for h in f['hasarlar']]
        tip = ', '.join(sorted({SINIF[dil].get(h['ham'], h['ham']) for h in hs})) or '—'
        satir.append([str(i), f"{a['marka']} {a['model']}", str(len(a['sonuc'])),
                      str(len(hs)), tip,
                      f"{sum(h['alan_yuzde'] for h in hs):.2f}"])
    tz = Table(satir, colWidths=[19*mm, 44*mm, 17*mm, 21*mm, 46*mm, 31*mm])
    tz.setStyle(TableStyle([
        ('FONTNAME',(0,0),(-1,0),'DejaVu-Bold'), ('FONTNAME',(0,1),(-1,-1),'DejaVu'),
        ('FONTSIZE',(0,0),(-1,-1),8.5), ('BACKGROUND',(0,0),(-1,0),LACI),
        ('TEXTCOLOR',(0,0),(-1,0),colors.white),
        ('ALIGN',(0,0),(0,-1),'CENTER'), ('ALIGN',(2,0),(3,-1),'CENTER'),
        ('ALIGN',(5,0),(5,-1),'CENTER'),
        ('GRID',(0,0),(-1,-1),0.4,CIZGI), ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),5), ('BOTTOMPADDING',(0,0),(-1,-1),5)] +
        [('BACKGROUND',(0,i),(-1,i),GRI) for i in range(2, len(satir), 2)]))
    st.append(tz)

    # ── ARAÇ BÖLÜMLERİ ──────────────────────────────────────────────────────
    for i, a in enumerate(veri, 1):
        st.append(PageBreak())
        st.append(Paragraph(T['s_arac'].format(n=i, ad=f"{a['marka']} {a['model']}"), S['h']))

        hs = [h for f in a['sonuc'] for h in f['hasarlar']]
        tb = Table([[T['b_marka'], P_(a['marka']), T['b_renk'],
                     a['renk_tr'] if dil == 'TR' else a['renk_en']],
                    [T['b_model'], P_(a['model']), T['b_foto'], str(len(a['sonuc']))],
                    ['', '', T['b_hasar'], str(len(hs))]],
                   colWidths=[26*mm, 52*mm, 36*mm, 54*mm])
        tb.setStyle(TableStyle([
            ('FONTNAME',(0,0),(-1,-1),'DejaVu'), ('FONTNAME',(0,0),(0,-1),'DejaVu-Bold'),
            ('FONTNAME',(2,0),(2,-1),'DejaVu-Bold'), ('FONTSIZE',(0,0),(-1,-1),8.5),
            ('BACKGROUND',(0,0),(0,-1),GRI), ('BACKGROUND',(2,0),(2,-1),GRI),
            ('GRID',(0,0),(-1,-1),0.4,CIZGI),
            ('TOPPADDING',(0,0),(-1,-1),4), ('BOTTOMPADDING',(0,0),(-1,-1),4)]))
        st.append(tb)

        veri_t = [[B_(T['t_foto']), B_(T['t_aci']), B_(T['t_parca']), B_(T['t_hasar']),
                   B_(T['t_alan']), B_(T['t_guven']), B_(T['t_durum'])]]
        for j, f in enumerate(a['sonuc'], 1):
            aci = f['aci_tr'] if dil == 'TR' else f['aci_en']
            if not f['hasarlar']:
                veri_t.append([str(j), aci, '—', '—', '—', '—', T['yok'].split('.')[0]])
                continue
            for h in f['hasarlar']:
                d = T['kesin'] if h['guven'] >= .70 else (
                    T['olasi'] if h['guven'] >= .45 else T['supheli'])
                veri_t.append([str(j), aci,
                               parca_ad(h['parca'], dil) if h['parca'] else T['atanmadi'],
                               SINIF[dil].get(h['ham'], h['ham']),
                               f"{h['alan_yuzde']:.2f}", f"{h['guven']*100:.1f}", d])
        tt = Table(veri_t, colWidths=[15*mm, 32*mm, 31*mm, 25*mm, 18*mm, 24*mm, 29*mm],
                   repeatRows=1)
        tt.setStyle(TableStyle([
            ('FONTNAME',(0,0),(-1,0),'DejaVu-Bold'), ('FONTNAME',(0,1),(-1,-1),'DejaVu'),
            ('FONTSIZE',(0,0),(-1,-1),8), ('BACKGROUND',(0,0),(-1,0),LACI),
            ('TEXTCOLOR',(0,0),(-1,0),colors.white),
            ('ALIGN',(0,0),(0,-1),'CENTER'), ('ALIGN',(4,0),(-1,-1),'CENTER'),
            ('GRID',(0,0),(-1,-1),0.4,CIZGI), ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
            ('TOPPADDING',(0,0),(-1,-1),4), ('BOTTOMPADDING',(0,0),(-1,-1),4)] +
            [('BACKGROUND',(0,r),(-1,r),GRI) for r in range(2, len(veri_t), 2)]))
        st.append(Spacer(1, 5)); st.append(tt)

        st.append(Paragraph(T['g_baslik'], S['h2']))
        satirlar, buf = [], []
        for f in a['sonuc']:
            g = _rapor_boyutu(
                SONUC / f'maskeli_{dil}' / f"{pathlib.Path(f['dosya']).stem}_{dil}.jpg")
            w, h = f['boyut']
            gw = 82*mm
            buf.append(RLImage(str(g), width=gw, height=gw*h/w))
            if len(buf) == 2:
                satirlar.append(buf); buf = []
        if buf:
            buf.append('')
            satirlar.append(buf)
        tg = Table(satirlar, colWidths=[86*mm, 86*mm])
        tg.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),
                                ('LEFTPADDING',(0,0),(-1,-1),1),
                                ('RIGHTPADDING',(0,0),(-1,-1),1),
                                ('TOPPADDING',(0,0),(-1,-1),3),
                                ('BOTTOMPADDING',(0,0),(-1,-1),3)]))
        st.append(tg)
        st.append(Paragraph(T['renk_k'], S['k']))

    # ── SİSTEM + NOTLAR ─────────────────────────────────────────────────────
    st.append(PageBreak())
    st.append(Paragraph(T['s_sistem'], S['h']))
    ts = Table([[T['m_parca'], SISTEM['parca']], [T['m_hasar'], SISTEM['hasar']],
                [T['m_map'], SISTEM['map50']], [T['m_sure'], SISTEM['sure']],
                [T['m_esik'], SISTEM['esik']], [T['m_coz'], SISTEM['coz']]],
               colWidths=[62*mm, 106*mm])
    ts.setStyle(TableStyle([
        ('FONTNAME',(0,0),(-1,-1),'DejaVu'), ('FONTNAME',(0,0),(0,-1),'DejaVu-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),8), ('BACKGROUND',(0,0),(0,-1),GRI),
        ('GRID',(0,0),(-1,-1),0.4,CIZGI),
        ('TOPPADDING',(0,0),(-1,-1),4), ('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    st.append(ts)

    st.append(Paragraph(T['s_not'], S['h']))
    st.append(Paragraph(T['yasal'], S['k']))

    doc.build(st, onFirstPage=dipnot, onLaterPages=dipnot)
    return dosya


if __name__ == '__main__':
    font_hazirla()
    veri = json.loads((SONUC / 'analiz_sonuc.json').read_text(encoding='utf-8'))
    for dil in ('TR', 'EN'):
        f = coklu_rapor(veri, dil, str(CIK / f'coklu_arac_raporu_{dil}.pdf'))
        print(f'{dil} -> {pathlib.Path(f).name}  '
              f'{pathlib.Path(f).stat().st_size//1024} KB')
