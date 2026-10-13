# -*- coding: utf-8 -*-
"""
HÜCRE 9 (v2) — ÇİFT DİLLİ HASAR RAPORU ÜRETİCİ  /  BILINGUAL DAMAGE REPORT
===========================================================================
DIL = 'TR'  -> TRAMER formatinda Turkce rapor  (tez / savunma icin)
DIL = 'EN'  -> ayni rapor Ingilizce            (uluslararasi yayin icin)

Model sinif adlari Turkce kalir (Cizik/Gocuk/Kirik); yalnizca GOSTERIM cevrilir.
Boylece model agirliklari ve defterlerle uyum bozulmaz.
"""
import os, sys, urllib.request
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image as RLImage)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage
import numpy as np

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  DİL SEÇİMİ / LANGUAGE                                                   ║
# ╚══════════════════════════════════════════════════════════════════════════╝
DIL = 'EN'          # 'TR' | 'EN'

# ─── Sinif adi esleme / class-name mapping ──────────────────────────────────
SINIF_AD = {
    'TR': {'Cizik': 'Çizik',   'Gocuk': 'Göçük', 'Kirik': 'Kırık'},
    'EN': {'Cizik': 'Scratch', 'Gocuk': 'Dent',  'Kirik': 'Crack'},
}

def parca_ad(ham: str, dil: str) -> str:
    """Carparts-Seg etiketleri zaten Ingilizce; TR'de Turkcelestirilir."""
    TR = {
        'back_bumper': 'Arka Tampon',      'back_door': 'Arka Kapı',
        'back_glass': 'Arka Cam',          'back_left_door': 'Sol Arka Kapı',
        'back_left_light': 'Sol Arka Far', 'back_light': 'Arka Far',
        'back_right_door': 'Sağ Arka Kapı','back_right_light': 'Sağ Arka Far',
        'front_bumper': 'Ön Tampon',       'front_door': 'Ön Kapı',
        'front_glass': 'Ön Cam',           'front_left_door': 'Sol Ön Kapı',
        'front_left_light': 'Sol Ön Far',  'front_light': 'Ön Far',
        'front_right_door': 'Sağ Ön Kapı', 'front_right_light': 'Sağ Ön Far',
        'hood': 'Kaput',                   'left_mirror': 'Sol Ayna',
        'object': 'Nesne',                 'right_mirror': 'Sağ Ayna',
        'tailgate': 'Bagaj Kapağı',        'trunk': 'Bagaj',
        'wheel': 'Tekerlek',
    }
    if dil == 'TR':
        return TR.get(ham, ham.replace('_', ' ').title())
    return ham.replace('_', ' ').title()

# ─── Tum arayuz metinleri / all UI strings ──────────────────────────────────
METIN = {
'TR': dict(
    baslik      = 'ARAÇ HASAR TESPİT RAPORU',
    altbaslik   = 'Yapay Zekâ Destekli Otomatik Ekspertiz Sistemi',
    belge_no    = 'Belge No',        tarih       = 'Düzenleme Tarihi',
    police      = 'Poliçe No',       rapor_turu  = 'Rapor Türü',
    on_tespit   = 'Otomatik Ön Tespit',
    b1          = '1. ARAÇ BİLGİLERİ',
    plaka='Plaka', marka='Marka', model='Model', yil='Model Yılı',
    sase='Şasi No', renk='Renk',
    b2          = '2. TESPİT EDİLEN HASARLAR',
    s_no='#', s_parca='Parça', s_hasar='Hasar Tipi',
    s_alan='Alan (%)', s_guven='Güven (%)', s_durum='Durum',
    kesin='Kesin', olasi='Olası', supheli='Şüpheli',
    ozet        = ('<b>Özet:</b> Toplam <b>{n}</b> hasar tespit edildi. '
                   'Etkilenen parçalar: <b>{p}</b>. Hasar tipleri: <b>{t}</b>. '
                   'Toplam hasarlı alan: <b>%{a:.2f}</b>.'),
    hasar_yok   = 'Görselde herhangi bir hasar tespit edilmemiştir.',
    b3          = '3. GÖRSEL ANALİZ',
    renk_kodu   = 'Renk kodları: {k}',
    b4          = '4. SİSTEM BİLGİSİ',
    m_parca='Parça Tespit Modeli', m_hasar='Hasar Tespit Modeli',
    m_map='Hasar Modeli mAP50',    m_sure='İşlem Süresi',
    m_coz='Görüntü Çözünürlüğü',   ms_gorsel='~{v:.0f} ms/görsel',
    yasal       = ('<b>ÖNEMLİ AÇIKLAMA:</b> Bu rapor yapay zekâ tabanlı görüntü analiz '
                   'sistemi tarafından otomatik olarak üretilmiştir ve <b>ön değerlendirme</b> '
                   'niteliğindedir. Nihai hasar tespiti, onarım kapsamı ve tazminat tutarı '
                   'yetkili eksper incelemesi sonucunda belirlenir. Sistem çıktıları yalnızca '
                   'eksper iş akışını hızlandırmak amacıyla referans olarak kullanılmalıdır. '
                   'Görüntü kalitesi, çekim açısı ve ışık koşulları tespit doğruluğunu '
                   'etkileyebilir.'),
    yok='—',
),
'EN': dict(
    baslik      = 'VEHICLE DAMAGE ASSESSMENT REPORT',
    altbaslik   = 'AI-Assisted Automatic Insurance Appraisal System',
    belge_no    = 'Document No',     tarih       = 'Issue Date',
    police      = 'Policy No',       rapor_turu  = 'Report Type',
    on_tespit   = 'Automated Preliminary Assessment',
    b1          = '1. VEHICLE DETAILS',
    plaka='Plate', marka='Make', model='Model', yil='Model Year',
    sase='VIN', renk='Colour',
    b2          = '2. DETECTED DAMAGE',
    s_no='#', s_parca='Part', s_hasar='Damage Type',
    s_alan='Area (%)', s_guven='Confidence (%)', s_durum='Status',
    kesin='Confirmed', olasi='Probable', supheli='Uncertain',
    ozet        = ('<b>Summary:</b> <b>{n}</b> damage instance(s) detected. '
                   'Affected parts: <b>{p}</b>. Damage types: <b>{t}</b>. '
                   'Total damaged area: <b>{a:.2f}%</b>.'),
    hasar_yok   = 'No damage was detected in the image.',
    b3          = '3. VISUAL ANALYSIS',
    renk_kodu   = 'Colour key: {k}',
    b4          = '4. SYSTEM INFORMATION',
    m_parca='Part Detection Model',  m_hasar='Damage Detection Model',
    m_map='Damage Model mAP@50',     m_sure='Inference Time',
    m_coz='Image Resolution',        ms_gorsel='~{v:.0f} ms/image',
    yasal       = ('<b>IMPORTANT NOTICE:</b> This report was generated automatically by an '
                   'AI-based image analysis system and constitutes a <b>preliminary '
                   'assessment</b>. The final damage assessment, repair scope and settlement '
                   'amount are determined by a qualified claims adjuster. System outputs '
                   'should be used for reference only, to accelerate the adjuster workflow. '
                   'Image quality, viewing angle and lighting conditions may affect detection '
                   'accuracy.'),
    yok='—',
),
}

# ─── Font (Turkce karakterler icin DejaVu) ──────────────────────────────────
def font_hazirla(font_dir='/content/fonts'):
    os.makedirs(font_dir, exist_ok=True)
    reg, bold = f'{font_dir}/DejaVuSans.ttf', f'{font_dir}/DejaVuSans-Bold.ttf'
    if not os.path.exists(reg):
        sistem = '/usr/share/fonts/truetype/dejavu'
        if os.path.exists(f'{sistem}/DejaVuSans.ttf'):
            import shutil
            shutil.copy(f'{sistem}/DejaVuSans.ttf', reg)
            shutil.copy(f'{sistem}/DejaVuSans-Bold.ttf', bold)
        else:
            base = ('https://raw.githubusercontent.com/dejavu-fonts/'
                    'dejavu-fonts/master/build/ttf')
            urllib.request.urlretrieve(f'{base}/DejaVuSans.ttf', reg)
            urllib.request.urlretrieve(f'{base}/DejaVuSans-Bold.ttf', bold)
    pdfmetrics.registerFont(TTFont('DejaVu', reg))
    pdfmetrics.registerFont(TTFont('DejaVu-Bold', bold))
    return reg, bold


# ╔══════════════════════════════════════════════════════════════════════════╗
def hasar_raporu_olustur(sonuc, img_np, arac_bilgi=None, dosya_adi=None,
                         dil=None, sistem_bilgi=None, hasar_renk=None,
                         rapor_dir='/content'):
    """
    Fuzyon sonucundan PDF rapor uretir.  /  Builds a PDF report from the fusion output.

    sonuc['birlesik'] ogeleri: parca (ham Ingilizce etiket), hasar (Cizik|Gocuk|Kirik),
                               alan_yuzde, guven, mask
    """
    dil = dil or DIL
    M = METIN[dil]
    SN = SINIF_AD[dil]
    arac_bilgi   = arac_bilgi or {}
    sistem_bilgi = sistem_bilgi or {}
    hasar_renk   = hasar_renk or {'Cizik': [1.0, .67, .24],
                                  'Gocuk': [.18, .66, .78],
                                  'Kirik': [.82, .29, .36]}

    simdi = datetime.now()
    on = 'HT' if dil == 'TR' else 'VD'
    belge_no  = arac_bilgi.get('hasar_no') or f"{on}-{simdi.strftime('%Y%m%d-%H%M%S')}"
    dosya_adi = dosya_adi or os.path.join(rapor_dir, f'report_{dil}_{belge_no}.pdf')

    ss = getSampleStyleSheet()
    st_b = ParagraphStyle('B', parent=ss['Title'], fontName='DejaVu-Bold', fontSize=17,
                          textColor=colors.HexColor('#0f172a'), spaceAfter=2, alignment=1)
    st_a = ParagraphStyle('A', parent=ss['Normal'], fontName='DejaVu', fontSize=9.5,
                          textColor=colors.HexColor('#64748b'), alignment=1, spaceAfter=12)
    st_h = ParagraphStyle('H', parent=ss['Heading2'], fontName='DejaVu-Bold', fontSize=11.5,
                          textColor=colors.HexColor('#1e40af'), spaceBefore=13, spaceAfter=6)
    st_p = ParagraphStyle('P', parent=ss['Normal'], fontName='DejaVu', fontSize=9, leading=13)
    st_k = ParagraphStyle('K', parent=ss['Normal'], fontName='DejaVu', fontSize=7.5,
                          textColor=colors.HexColor('#64748b'), leading=11)

    doc = SimpleDocTemplate(dosya_adi, pagesize=A4, leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=15*mm, bottomMargin=15*mm)
    story = [Paragraph(M['baslik'], st_b), Paragraph(M['altbaslik'], st_a)]

    KUTU = TableStyle([
        ('FONTNAME',(0,0),(-1,-1),'DejaVu'),
        ('FONTNAME',(0,0),(0,-1),'DejaVu-Bold'),
        ('FONTNAME',(2,0),(2,-1),'DejaVu-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),8.5),
        ('BACKGROUND',(0,0),(0,-1),colors.HexColor('#f1f5f9')),
        ('BACKGROUND',(2,0),(2,-1),colors.HexColor('#f1f5f9')),
        ('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#cbd5e1')),
        ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
        ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)])

    t0 = Table([[M['belge_no'], belge_no, M['tarih'], simdi.strftime('%d.%m.%Y %H:%M')],
                [M['police'], arac_bilgi.get('police_no', M['yok']),
                 M['rapor_turu'], M['on_tespit']]],
               colWidths=[32*mm, 46*mm, 38*mm, 46*mm])
    t0.setStyle(KUTU); story.append(t0)

    story.append(Paragraph(M['b1'], st_h))
    t1 = Table([[M['plaka'], arac_bilgi.get('plaka', M['yok']),
                 M['marka'], arac_bilgi.get('marka', M['yok'])],
                [M['model'], arac_bilgi.get('model', M['yok']),
                 M['yil'],   str(arac_bilgi.get('yil', M['yok']))],
                [M['sase'],  arac_bilgi.get('sase', M['yok']),
                 M['renk'],  arac_bilgi.get('renk', M['yok'])]],
               colWidths=[32*mm, 46*mm, 38*mm, 46*mm])
    t1.setStyle(KUTU); story.append(t1)

    story.append(Paragraph(M['b2'], st_h))
    birlesik = sonuc['birlesik']
    if birlesik:
        veri = [[M['s_no'], M['s_parca'], M['s_hasar'],
                 M['s_alan'], M['s_guven'], M['s_durum']]]
        for i, b in enumerate(birlesik, 1):
            d = M['kesin'] if b['guven'] >= .70 else (M['olasi'] if b['guven'] >= .45
                                                      else M['supheli'])
            veri.append([str(i), parca_ad(b['parca'], dil), SN.get(b['hasar'], b['hasar']),
                         f"{b['alan_yuzde']:.2f}", f"{b['guven']*100:.1f}", d])
        th = Table(veri, colWidths=[10*mm, 38*mm, 32*mm, 26*mm, 26*mm, 30*mm])
        stil = [('FONTNAME',(0,0),(-1,0),'DejaVu-Bold'),
                ('FONTNAME',(0,1),(-1,-1),'DejaVu'),
                ('FONTSIZE',(0,0),(-1,-1),8.5),
                ('BACKGROUND',(0,0),(-1,0),colors.HexColor('#1e40af')),
                ('TEXTCOLOR',(0,0),(-1,0),colors.white),
                ('ALIGN',(0,0),(0,-1),'CENTER'), ('ALIGN',(3,0),(-1,-1),'CENTER'),
                ('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#cbd5e1')),
                ('VALIGN',(0,0),(-1,-1),'MIDDLE'),
                ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]
        for i in range(1, len(veri)):
            if i % 2 == 0:
                stil.append(('BACKGROUND',(0,i),(-1,i),colors.HexColor('#f8fafc')))
        th.setStyle(TableStyle(stil)); story.append(th)

        ta = sum(b['alan_yuzde'] for b in birlesik)
        ps = ', '.join(sorted({parca_ad(b['parca'], dil) for b in birlesik}))
        ts = ', '.join(sorted({SN.get(b['hasar'], b['hasar']) for b in birlesik}))
        story.append(Spacer(1, 7))
        story.append(Paragraph(M['ozet'].format(n=len(birlesik), p=ps, t=ts, a=ta), st_p))
    else:
        story.append(Paragraph(M['hasar_yok'], st_p))

    story.append(Paragraph(M['b3'], st_h))
    vis = img_np.astype(np.float32).copy()
    for b in birlesik:
        c = np.array(hasar_renk.get(b['hasar'], [.5, .5, .5])) * 255
        vis[b['mask']] = vis[b['mask']] * .42 + c * .58
    vis = vis.astype(np.uint8)
    tmp = os.path.join(rapor_dir, '_rapor_gorsel.jpg')
    PILImage.fromarray(vis).save(tmp, 'JPEG', quality=88)
    ih, iw = vis.shape[:2]
    oran = min(160*mm/iw, 105*mm/ih)
    story.append(RLImage(tmp, width=iw*oran, height=ih*oran))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        M['renk_kodu'].format(k=' | '.join(SN.get(k, k) for k in hasar_renk)), st_k))

    story.append(Paragraph(M['b4'], st_h))
    t4 = Table([[M['m_parca'], sistem_bilgi.get('parca_model', 'YOLOv11m-seg (Carparts-Seg)')],
                [M['m_hasar'], sistem_bilgi.get('hasar_model', M['yok'])],
                [M['m_map'],   f"{sistem_bilgi.get('mask_mAP50', 0):.4f}"],
                [M['m_sure'],  M['ms_gorsel'].format(v=sistem_bilgi.get('inference_ms', 0))],
                [M['m_coz'],   f"{sonuc['goruntu_boyut'][0]} x {sonuc['goruntu_boyut'][1]} px"]],
               colWidths=[58*mm, 104*mm])
    t4.setStyle(TableStyle([
        ('FONTNAME',(0,0),(-1,-1),'DejaVu'), ('FONTNAME',(0,0),(0,-1),'DejaVu-Bold'),
        ('FONTSIZE',(0,0),(-1,-1),8),
        ('BACKGROUND',(0,0),(0,-1),colors.HexColor('#f8fafc')),
        ('GRID',(0,0),(-1,-1),0.4,colors.HexColor('#cbd5e1')),
        ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    story.append(t4)
    story.append(Spacer(1, 10))
    story.append(Paragraph(M['yasal'], st_k))

    doc.build(story)
    return dosya_adi
