"""Check the delivered document and worked examples; no model dependencies."""
from pathlib import Path
from fractions import Fraction as F
import hashlib
import json
import math
from pypdf import PdfReader
from content import PAGES

ROOT=Path(__file__).resolve().parents[3]
pdf=ROOT/'output/pdf/lens-mathematics-a-grounded-guide.pdf'
r=PdfReader(pdf)
assert len(r.pages)==len(PAGES)==18
assert len(r.outline)==18
assert not r.is_encrypted
for i,(page,(_,title,_)) in enumerate(zip(r.pages,PAGES),1):
    text=page.extract_text()
    assert title in text.replace('\n',' '),(i,title)
    assert f'{i} / 18' in text
    assert '\ufffd' not in text and '\u25a0' not in text
    assert float(page.mediabox.width)==420 and float(page.mediabox.height)==780
links=[]
for page in r.pages:
    for a in page.get('/Annots',[]):
        obj=a.get_object()
        action=obj.get('/A',{})
        if action.get('/S')=='/URI': links.append(action['/URI'])
assert len(links)==5 and all(u.startswith('https://') for u in links)
A,B,dA,dB=F(2),F(3),F(1,10),F(-1,5)
predicted=(F(1,2)*B*dA+F(1,2)*A*dB,F(1,5)*dB)
actual=(F(1,2)*(A+dA)*(B+dB)-F(1,2)*A*B,F(1,5)*dB)
assert predicted==(F(-1,20),F(-1,25)) and actual==(F(-3,50),F(-1,25))
assert actual[0]-predicted[0]==F(-1,100)
def mm(z): return z/(1+z)
for h in (F(1,10),F(1,20),F(1,10000)):
    d=(mm(1+h)-mm(1-h))/(2*h)
    assert d==1/(4-h*h)
assert round(float(mm(1-F(1,10000))),2)==.5
assert round(float(mm(1+F(1,10000))),2)==.5
assert math.isclose(.01*math.sqrt(128*2560),5.724334022399462,rel_tol=1e-12)
assert F(1,128)==F('0.0078125')
layout=json.loads((ROOT/'tmp/pdfs/lens-math-guide/layout.json').read_text())
assert all(p['bottom_of_content']>=48 for p in layout)
report={'status':'passed','basis':'document-and-illustrative-mathematics-verification',
    'pages':len(r.pages),'bookmarks':len(r.outline),'external_links':links,
    'page_size_points':[420,780],'body_font_points':13.2,
    'pdf_bytes':pdf.stat().st_size,'pdf_sha256':hashlib.sha256(pdf.read_bytes()).hexdigest(),
    'checks':['page count and bookmarks','all expected titles and page footers','HTTPS reference links',
        'no replacement or black-square text glyphs','strict content/footer separation',
        'reaction-rate example using exact rational arithmetic','central-difference formula using exact rational arithmetic',
        'rounded readout example','global RMS multiplier and bf16 spacing'],
    'visual_qa':'All 18 pages inspected in rendered contact sheets; repaired equation asset collision and inspected page 8 again at higher resolution.',
    'independent_review':'Mathematical manuscript review passed; its direction-notation and layer-label clarifications were applied.',
    'model_execution':False}
(Path(__file__).parent/'VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
