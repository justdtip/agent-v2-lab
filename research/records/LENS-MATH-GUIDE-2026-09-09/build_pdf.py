"""Build an explicitly paginated, phone-readable PDF. Fails on page overflow."""
from pathlib import Path
import json
import os
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle, Image
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor, white
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from PIL import Image as PILImage
from content import PAGES

ROOT=Path(__file__).resolve().parents[3]
ASSETS=ROOT/'tmp/pdfs/lens-math-guide/assets'
OUT=ROOT/'output/pdf/lens-mathematics-a-grounded-guide.pdf'
FONTS=Path('/private/tmp/codex-gpu-analysis-plot-deps/matplotlib/mpl-data/fonts/ttf')
for name,file in [('Body','DejaVuSans.ttf'),('Bold','DejaVuSans-Bold.ttf'),('Italic','DejaVuSans-Oblique.ttf'),('BoldItalic','DejaVuSans-BoldOblique.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(FONTS/file)))
pdfmetrics.registerFontFamily('Body',normal='Body',bold='Bold',italic='Italic',boldItalic='BoldItalic')
INK=HexColor('#20313B'); TEAL=HexColor('#287B8E'); GRAY=HexColor('#53616B'); LIGHT=HexColor('#EDF5F6')
W,H=420,780
M=31
CW=W-2*M
styles={
'body':ParagraphStyle('body',fontName='Body',fontSize=13.2,leading=18.3,textColor=INK,spaceAfter=0),
'small':ParagraphStyle('small',fontName='Body',fontSize=10.4,leading=14.2,textColor=GRAY),
'caption':ParagraphStyle('caption',fontName='Body',fontSize=11.1,leading=15.0,textColor=GRAY),
'heading':ParagraphStyle('heading',fontName='Bold',fontSize=23,leading=27,textColor=INK),
'box':ParagraphStyle('box',fontName='Body',fontSize=12.6,leading=17.5,textColor=INK),
'cell':ParagraphStyle('cell',fontName='Body',fontSize=11.3,leading=15.0,textColor=INK),
}
c=canvas.Canvas(str(OUT),pagesize=(W,H),pageCompression=1,invariant=1)
c.setTitle('From reaction rates to model lenses: a grounded mathematical guide')
c.setAuthor('Codex - prepared for Daniel')
c.setSubject('Vectors, Jacobians, finite differences and the lens precision diagnosis')
c.setKeywords('linear algebra, Jacobian, interpretability, finite differences, bfloat16, chemistry')
c.setViewerPreference('DisplayDocTitle','true')
layouts=[]
y=0


def consume(height,label):
    global y
    if y-height < 48 and not os.environ.get('PDF_LAYOUT_PROBE'):
        raise RuntimeError(f'Page {page}: {label} would overflow; y={y:.1f}, height={height:.1f}')
    y-=height


def para(text,style='body',gap=10,x=M,width=CW):
    global y
    p=Paragraph(text,styles[style]); _,hh=p.wrap(width,H)
    consume(hh,'paragraph: '+text[:45])
    p.drawOn(c,x,y)
    consume(gap,'paragraph gap')


def picture(key,maxheight):
    global y
    filename=ASSETS/(key+'.png')
    with PILImage.open(filename) as im:
        iw,ih=im.size
    if maxheight is None:
        scale=min(CW/iw,72/260)
    else:
        scale=min(CW/iw,maxheight/ih)
    ww,hh=iw*scale,ih*scale
    consume(hh,'image '+key)
    Image(str(filename),width=ww,height=hh).drawOn(c,M+(CW-ww)/2,y)
    consume(9,'image gap')


for page,(kicker,title,blocks) in enumerate(PAGES,1):
    c.bookmarkPage('p'+str(page))
    c.addOutlineEntry(title,'p'+str(page),level=0,closed=False)
    c.setFillColor(TEAL); c.rect(M,H-27,27,3,fill=1,stroke=0)
    c.setFillColor(GRAY); c.setFont('Body',8.6)
    c.drawString(M,H-45,kicker.upper())
    y=H-62
    para(title,'heading',gap=16)
    for block in blocks:
        kind=block[0]
        if kind=='p': para(block[1])
        elif kind=='small': para(block[1],'small',gap=9)
        elif kind=='eq':
            picture(block[1],None)
            para(block[2],'caption',gap=13)
        elif kind=='figure':
            picture(block[1],block[2]); para(block[3],'small',gap=11)
        elif kind=='callout':
            body=Paragraph('<b>'+block[1]+'</b><br/>'+block[2],styles['box'])
            _,hh=body.wrap(CW-22,H)
            consume(hh+20,'callout')
            c.setFillColor(LIGHT);c.roundRect(M,y,CW,hh+20,5,stroke=0,fill=1)
            body.drawOn(c,M+11,y+10)
            consume(12,'callout gap')
        elif kind=='table':
            header,rows,widths=block[1:]
            data=[[Paragraph('<b>'+t+'</b>',styles['cell']) for t in header]]+[[Paragraph(t,styles['cell']) for t in row] for row in rows]
            t=Table(data,colWidths=widths,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),LIGHT),('VALIGN',(0,0),(-1,-1),'TOP'),
                ('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),
                ('LINEBELOW',(0,0),(-1,0),.6,TEAL),('LINEBELOW',(0,-1),(-1,-1),.5,HexColor('#CDD5D9'))]))
            _,hh=t.wrap(CW,H);consume(hh,'table');t.drawOn(c,M,y);consume(12,'table gap')
        elif kind=='step':
            para('<b>'+block[1]+'.</b> '+block[2],gap=13)
        elif kind=='gloss':
            para('<b>'+block[1]+'</b><br/>'+block[2],style='box',gap=12)
        elif kind=='link':
            para('<link href="'+block[2]+'" color="#287B8E"><u>'+block[1]+'</u></link>','small',gap=12)
        else: raise ValueError(kind)
    layouts.append({'page':page,'title':title,'bottom_of_content':round(y,2)})
    c.setStrokeColor(HexColor('#DDE5E8'));c.setLineWidth(.5);c.line(M,35,W-M,35)
    c.setFont('Body',8);c.setFillColor(GRAY)
    c.drawString(M,22,'LENS MATHEMATICS  /  9 SEP 2026')
    c.drawRightString(W-M,22,f'{page} / {len(PAGES)}')
    c.showPage()
c.save()
(ROOT/'tmp/pdfs/lens-math-guide/layout.json').write_text(json.dumps(layouts,indent=2)+'\n')
print('Created',OUT,'with',len(PAGES),'pages')
