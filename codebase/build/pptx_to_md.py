import os
import re
import zipfile
import argparse
from xml.etree import ElementTree as ET


A = '{http://schemas.openxmlformats.org/drawingml/2006/main}'
P = '{http://schemas.openxmlformats.org/presentationml/2006/main}'


def slide_number(name):
    return int(re.findall(r'\d+', name.rsplit('/', 1)[-1])[0])


def paragraphs(root):
    lines = []
    for sp in root.iter(P + 'sp'):
        for p in sp.iter(A + 'p'):
            text = ''.join(r.text or '' for r in p.iter(A + 't')).strip()
            # '‹#›' is the slide-number placeholder
            if not text or text == '‹#›':
                continue
            ppr = p.find(A + 'pPr')
            level = int(ppr.get('lvl', 0)) if ppr is not None else 0
            lines.append((level, text))
    return lines


def tables(root):
    out = []
    for tbl in root.iter(A + 'tbl'):
        rows = [[''.join(t.text or '' for t in tc.iter(A + 't')) for tc in tr.iter(A + 'tc')]
                for tr in tbl.iter(A + 'tr')]
        md = ['| ' + ' | '.join(rows[0]) + ' |', '|' + '---|' * len(rows[0])]
        md += ['| ' + ' | '.join(r) + ' |' for r in rows[1:]]
        out.append('\n'.join(md))
    return out


def convert(args):
    z = zipfile.ZipFile(args.input_file)
    names = z.namelist()
    slides = sorted([n for n in names if re.fullmatch(r'ppt/slides/slide\d+\.xml', n)], key=slide_number)
    media_dir = os.path.splitext(args.output_file)[0] + '_media'
    md = [f'# {args.title}\n', f'Converted from `{args.input_file}`.\n']

    for n in slides:
        num = slide_number(n)
        lines = paragraphs(ET.fromstring(z.read(n)))
        title = lines[0][1] if lines else ''
        md.append(f'## Slide {num}: {title}\n')
        for level, text in lines[1:]:
            md.append('  ' * level + '- ' + text)
        md += [''] + tables(ET.fromstring(z.read(n)))

        rels = f'ppt/slides/_rels/slide{num}.xml.rels'
        rels = z.read(rels).decode() if rels in names else ''
        images = re.findall(r'Target="\.\./media/([^"]+)"', rels)
        for img in images:
            os.makedirs(media_dir, exist_ok=True)
            out_name = f'slide{num:02d}_{img}'
            with open(os.path.join(media_dir, out_name), 'wb') as f:
                f.write(z.read('ppt/media/' + img))
            md.append(f'![slide {num}]({os.path.basename(media_dir)}/{out_name})')

        for notes in re.findall(r'Target="\.\./notesSlides/([^"]+)"', rels):
            text = ' '.join(t for _, t in paragraphs(ET.fromstring(z.read('ppt/notesSlides/' + notes))))
            if text:
                md.append(f'\n> Speaker note: {text}')
        md.append('')

    with open(args.output_file, 'w') as f:
        f.write(re.sub(r'\n{3,}', '\n\n', '\n'.join(md)).rstrip() + '\n')
    print(f'{len(slides)} slides -> {args.output_file}')


def parse_args():
    parser = argparse.ArgumentParser(description='Convert a .pptx to markdown; images go to <output>_media/')
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    parser.add_argument('--title', type=str, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    convert(parse_args())
