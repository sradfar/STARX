"""Copy only paper figures/captions and geographic evidence; no document edits."""
import json
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import zipfile

APP = Path(__file__).resolve().parents[1]
WORK = APP.parent
out = APP/'assets/paper'
out.mkdir(exist_ok=True)
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
      'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
captions = []
with zipfile.ZipFile(WORK/'Manuscript/npj_DT R01.docx') as z:
    root = ET.fromstring(z.read('word/document.xml'))
    rel = {r.attrib['Id']: r.attrib['Target'] for r in ET.fromstring(z.read('word/_rels/document.xml.rels'))}
    pending = None
    for p in root.findall('w:body/w:p', ns):
        # Ignore tracked deletions, including superseded embedded figure versions.
        parent = {c: n for n in p.iter() for c in n}
        def live(node):
            while node is not p:
                if node.tag in (f'{{{ns["w"]}}}del', f'{{{ns["w"]}}}moveFrom'):
                    return False
                node = parent.get(node, p)
            return True
        blips = [b for b in p.findall('.//a:blip', ns) if live(b)]
        if blips:
            pending = rel[blips[-1].attrib[f'{{{ns["r"]}}}embed']]
        text = ''.join(t.text or '' for t in p.findall('.//w:t', ns) if live(t)).strip()
        if text.startswith('Figure ') and len(text) > 10 and text[7].isdigit() and pending:
            number = int(text.split('.')[0].split()[1])
            if number >= 2:
                target = 'word/'+pending.lstrip('/') if not pending.startswith('/') else pending.lstrip('/')
                ext = Path(target).suffix
                filename = f'figure_{number}{ext}'
                (out/filename).write_bytes(z.read(target))
                captions.append(dict(number=number, file=filename, caption=text.replace('Full-local', 'Long-record local').replace('full-local', 'long-record local')))
            pending = None
(out/'figures.json').write_text(json.dumps(captions, indent=2))
shutil.copy2(WORK/'npj_runs/manuscript_package_v1/tables/table_1_station_context.csv', APP/'assets/data/station_context.csv')
shutil.copy2(WORK/'npj_runs/manuscript_package_v1/tables/table_3_geographic_mechanisms.csv', APP/'assets/data/geographic_mechanisms.csv')
print(f'Prepared {len(captions)} current main-text figure views and geographic evidence.')
