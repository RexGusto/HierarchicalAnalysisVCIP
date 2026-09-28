import os
import argparse

import pandas as pd
from pypdf import PdfReader


def read_pdf_page(fp, page_number, output_file, pdfminer=False):
    if pdfminer:
        from pdfminer import high_level
        text = high_level.extract_text(fp, "", [page_number])
    else:
        reader = PdfReader(fp)

        print(len(reader.pages))
        page = reader.pages[page_number]

        text = page.extract_text()

    print(text)

    print(output_file)
    with open(output_file, 'w', encoding='utf-8') as f:
       f.write(text)

    return text


def save_pdf_as_csv(text, output_file):
    lines = text.split('\n')

    table_data = [line.split(' ') for line in lines]

    df = pd.DataFrame(table_data)
    df.to_csv(output_file, index=False, header=False)
    return 0


def parse_args():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--input_file', type=str,
                        default=os.path.join('deprecated', 'old', 'papers', 'which_tokens.pdf'),
                        help='paper pdf title')
    parser.add_argument('--pdf_page_number', type=int, default=4,
                        help='pdf page number starting from 0')
    parser.add_argument('--pdfminer', action='store_true')

    # output
    parser.add_argument('--results_dir', type=str,
                        default=os.path.join('results_all', 'extracted_text'),
                        help='The directory where results will be stored')

    args= parser.parse_args()
    return args


def main():
    args = parse_args()
    os.makedirs(args.results_dir, exist_ok=True)
    fn = os.path.splitext(os.path.split(args.input_file)[-1])[0]
    output_file = f'{fn}_{args.pdf_page_number}'
    args.output_file = os.path.join(args.results_dir, output_file)

    text = read_pdf_page(args.input_file, args.pdf_page_number,
                         f'{args.output_file}.txt', args.pdfminer)

    save_pdf_as_csv(text, f'{args.output_file}.csv')

    return 0


if __name__ == '__main__':
    main()
