"""Build a phone-readable PDF from the audited account and a vector plot of the saved counts."""
import hashlib
import html
import json
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.graphics.shapes import Drawing, String, Line
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics import renderPDF
import pypdfium2 as pdfium
from PIL import Image, ImageDraw
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
pdfmetrics.registerFont(TTFont('Arial', '/System/Library/Fonts/Supplemental/Arial.ttf'))
pdfmetrics.registerFont(TTFont('ArialB', '/System/Library/Fonts/Supplemental/Arial Bold.ttf'))
pdfmetrics.registerFontFamily('Arial', normal='Arial', bold='ArialB', italic='Arial', boldItalic='ArialB')
INK = colors.HexColor('#173247')
BLUE = colors.HexColor('#256F9C')
ORANGE = colors.HexColor('#BA592E')
MUTED = colors.HexColor('#586776')
PAGE = (420, 595)
WIDTH = 364


def inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r'\[([^\]]+)\]\((https?://[^)]+)\)', r'<link href="\2" color="#256F9C">\1</link>', s)
    s = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'`([^`]+)`', r'<font name="Courier">\1</font>', s)
    return s


styles = {
    'body': ParagraphStyle('body', fontName='Arial', fontSize=10.5, leading=14.0, textColor=INK,
                           spaceAfter=8, splitLongWords=True),
    'title': ParagraphStyle('title', fontName='ArialB', fontSize=22, leading=25, textColor=INK,
                            spaceAfter=14, keepWithNext=True),
    'h2': ParagraphStyle('h2', fontName='ArialB', fontSize=14.5, leading=18, textColor=INK,
                         spaceBefore=13, spaceAfter=8, keepWithNext=True),
    'h3': ParagraphStyle('h3', fontName='ArialB', fontSize=11.7, leading=15, textColor=INK,
                         spaceBefore=8, spaceAfter=6, keepWithNext=True),
    'small': ParagraphStyle('small', fontName='Arial', fontSize=8.4, leading=11, textColor=MUTED,
                            spaceAfter=7),
    'cell': ParagraphStyle('cell', fontName='Arial', fontSize=8.5, leading=11, textColor=INK),
    'bullet': ParagraphStyle('bullet', fontName='Arial', fontSize=10.5, leading=14.0, textColor=INK,
                             leftIndent=11, firstLineIndent=-8, spaceAfter=7),
    'quote': ParagraphStyle('quote', fontName='Arial', fontSize=10.5, leading=14.0, textColor=INK,
                            leftIndent=10, rightIndent=7, borderColor=BLUE, borderWidth=0.7,
                            borderPadding=8, spaceBefore=5, spaceAfter=12),
}


def chart(report):
    d = Drawing(WIDTH, 416)
    d.add(String(0, 402, 'Next-token agreement on the recorded trace', fontName='ArialB', fontSize=12, fillColor=INK))
    d.add(String(0, 386, 'Same 96 text tokens; one selected prompt; descriptive counts only', fontName='Arial', fontSize=8.2, fillColor=MUTED))
    for j, m in enumerate(report['models']):
        y = 229 if j == 0 else 47
        name = '4B: its own reported greedy continuation' if j == 0 else '12B: teacher-forced on the 4B continuation'
        d.add(String(0, y+126, name, fontName='ArialB', fontSize=10.2, fillColor=INK))
        plot = LinePlot()
        plot.x, plot.y, plot.width, plot.height = 34, y, 317, 111
        plot.data = [tuple((r['layer'], r[method]['rank1']) for r in m['by_layer'])
                     for method in ('lens','logit_lens')]
        plot.xValueAxis.valueMin = 1
        plot.xValueAxis.valueMax = m['layer_max']
        plot.xValueAxis.valueSteps = [1, 8, 16, 24, 33] if j == 0 else [1, 12, 24, 36, 47]
        plot.yValueAxis.valueMin, plot.yValueAxis.valueMax = 0, 96
        plot.yValueAxis.valueSteps = [0, 24, 48, 72, 96]
        plot.xValueAxis.labels.fontName = plot.yValueAxis.labels.fontName = 'Arial'
        plot.xValueAxis.labels.fontSize = plot.yValueAxis.labels.fontSize = 8
        plot.lines[0].strokeColor, plot.lines[1].strokeColor = BLUE, ORANGE
        plot.lines[0].strokeWidth = plot.lines[1].strokeWidth = 1.7
        plot.yValueAxis.visibleGrid = True
        plot.yValueAxis.gridStrokeColor = colors.HexColor('#E0E5E9')
        plot.yValueAxis.gridStrokeWidth = 0.35
        d.add(plot)
        d.add(String(154,y-24,'Displayed source layer',fontName='Arial',fontSize=8.3,fillColor=MUTED))
    d.add(Line(25,13,43,13,strokeColor=BLUE,strokeWidth=2))
    d.add(String(49,10,'Fitted Jacobian lens',fontName='Arial',fontSize=8.4,fillColor=INK))
    d.add(Line(206,13,224,13,strokeColor=ORANGE,strokeWidth=2))
    d.add(String(230,10,'Plain logit lens',fontName='Arial',fontSize=8.4,fillColor=INK))
    return d


def footer(canvas, doc):
    canvas.setStrokeColor(colors.HexColor('#DCE2E7'))
    canvas.line(28,27,392,27)
    canvas.setFont('Arial',7.5); canvas.setFillColor(MUTED)
    canvas.drawString(28,16,'PIRATE GOLDFISH  |  Exploratory claims audit  |  10 September 2026')
    canvas.drawRightString(392,16,str(doc.page))


def markdown_story(text):
    lines = text.splitlines(); out = []; i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line: i += 1; continue
        if line.startswith('|'):
            rows=[]
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells=[x.strip() for x in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[-: ]+',x) for x in cells): rows.append(cells)
                i += 1
            n=len(rows[0]); assert all(len(r)==n for r in rows)
            first=0.36 if n >= 4 else 0.25
            widths=[WIDTH*first]+[WIDTH*(1-first)/(n-1)]*(n-1)
            data=[[Paragraph(('<b>'+inline(x)+'</b>') if ri==0 else inline(x),styles['cell'])
                   for x in row] for ri,row in enumerate(rows)]
            t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
            t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7EFF4')),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),
                ('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),
                ('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,-1),0.4,colors.HexColor('#D9E0E5'))]))
            out.extend([t,Spacer(1,10)]);continue
        if line.startswith('# '):
            out.append(Paragraph(inline(line[2:]),styles['title']));i+=1;continue
        if line.startswith('## '):
            out.append(Paragraph(inline(line[3:]),styles['h2']));i+=1;continue
        if line.startswith('### '):
            out.append(Paragraph(inline(line[4:]),styles['h3']));i+=1;continue
        if line.startswith('* '):
            out.append(Paragraph('• '+inline(line[2:]),styles['bullet']));i+=1;continue
        if line.startswith('> '):
            out.append(Paragraph(inline(line[2:]),styles['quote']));i+=1;continue
        para=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].strip().startswith(('#','|','* ','> ')):
            para.append(lines[i].strip());i+=1
        content = ' '.join(para)
        if content.startswith('**Finding, technique, implementation:**'):
            out.extend([PageBreak(), Paragraph('Audit and reproducibility', styles['h2'])])
        style = styles['body']
        out.append(Paragraph(inline(content), style))
    return out


def main():
    report=json.loads((HERE/'analysis.json').read_text())
    drawing=chart(report)
    renderPDF.drawToFile(drawing,str(HERE/'readout-agreement.pdf'))
    drawing.scale(0.90, 0.90)
    drawing.width *= 0.90
    drawing.height *= 0.90
    story=[Paragraph('What the picture can tell us',styles['title']),
           Paragraph('One trace can reveal a useful pattern without establishing a general mechanism. This report separates the saved numbers, the interpretation they support, and the claims they leave open.',styles['body']),
           drawing,
           Paragraph('Vertical scale: number of scored text tokens ranked first, out of 96. The 12B did not generate the text. These curves measure a readout target; they do not measure workspace size or reasoning quality.',styles['small']),
           PageBreak()]
    story.extend(markdown_story((HERE/'README.md').read_text()))
    verification = json.loads((HERE/'verification.json').read_text())
    story.append(Paragraph('What was checked', styles['h3']))
    checks = [
        '15,552 recorded score cells checked for array length and basic numerical consistency.',
        'Independent formulations reproduce every first-place count and reversal count.',
        f"All {len(verification['source_corruptions_refused'])} preserved source corruptions and all {len(verification['array_mutations_refused'])} malformed-array controls are refused.",
        'The calculations reproduce after copying the record to another directory.',
        'No model, tokenizer, matrix or checkpoint was loaded; no experimental workspace gate is certified.'
    ]
    for item in checks: story.append(Paragraph('• '+item, styles['bullet']))
    story.append(Paragraph('Source visualization SHA-256: 12f2a28ecc94c61f5b3965a0a4989abc389b45bd470cb0cf6e628199a4b245c4', styles['small']))
    doc=SimpleDocTemplate(str(HERE/'pirate-goldfish-claims.pdf'),pagesize=PAGE,leftMargin=28,rightMargin=28,
                          topMargin=27,bottomMargin=37,title='Pirate Goldfish: What We Can Claim',author='Codex')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    output=HERE/'pirate-goldfish-claims.pdf'
    pages=PdfReader(output).pages
    assert len(pages)>1
    text='\n'.join(p.extract_text() for p in pages)
    for required in ['76/96','68/96','201','70','goldfish','teacher-forced','pre-registration']:
        assert required in text,required
    assert '\u25a0' not in text
    qa=HERE/'qa';qa.mkdir(exist_ok=True)
    for old in qa.glob('page-*.png'): old.unlink()
    pdf=pdfium.PdfDocument(str(output))
    thumbs=[]
    for i,page in enumerate(pdf):
        img=page.render(scale=1.4).to_pil().convert('RGB')
        img.save(qa/f'page-{i+1:02d}.png')
        thumb=img.copy();thumb.thumbnail((210,298))
        card=Image.new('RGB',(224,320),'#e6e9eb');card.paste(thumb,((224-thumb.width)//2,5))
        ImageDraw.Draw(card).text((10,304),str(i+1),fill='black');thumbs.append(card)
    cols=3;rows=(len(thumbs)+cols-1)//cols
    sheet=Image.new('RGB',(cols*224,rows*320),'white')
    for i,img in enumerate(thumbs):sheet.paste(img,((i%cols)*224,(i//cols)*320))
    sheet.save(qa/'contact-sheet.png')
    (HERE/'pdf-check.json').write_text(json.dumps({'pages':len(pages),'text_characters':len(text),
        'required_text_present':True,'pdf_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
        'page_rendering':'pypdfium2, all pages, 1.4x','visual_review':'pending',
        'input_analysis_sha256':hashlib.sha256((HERE/'analysis.json').read_bytes()).hexdigest(),
        'input_readme_sha256':hashlib.sha256((HERE/'README.md').read_bytes()).hexdigest()},indent=2)+'\n')
    print({'pages':len(pages),'pdf_bytes':output.stat().st_size,'contact_sheet':str(qa/'contact-sheet.png')})


if __name__=='__main__':main()
