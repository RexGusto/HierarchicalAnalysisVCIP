import os
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Replace one serial's rows in committed CSVs with a fresh download; the old rows go to --archive_dir")
    parser.add_argument('--serial', type=int, required=True)
    parser.add_argument('--fresh_file', type=str, required=True, help='new download holding the serial')
    parser.add_argument('--targets', nargs='+', required=True, help='CSVs in data/ whose rows of the serial are replaced')
    parser.add_argument('--datasets', nargs='+', default=None, help='only replace rows of these dataset_name values')
    parser.add_argument('--archive_dir', type=str, required=True)
    parser.add_argument('--execute', action='store_true', help='write; otherwise only report')
    args = parser.parse_args()

    fresh = pd.read_csv(args.fresh_file, low_memory=False)
    fresh = fresh[fresh.serial == args.serial]
    if args.datasets:
        fresh = fresh[fresh.dataset_name.isin(args.datasets)]
    os.makedirs(args.archive_dir, exist_ok=True)

    for fp in args.targets:
        df = pd.read_csv(fp, low_memory=False)
        old = (df.serial == args.serial) & (df.dataset_name.isin(args.datasets) if args.datasets else True)
        new = fresh.reindex(columns=df.columns)
        print(f'{fp}: {int(old.sum())} old rows -> {len(new)} new rows; {len(set(fresh.columns) - set(df.columns))} fresh columns dropped')
        if args.execute:
            df[old].to_csv(os.path.join(args.archive_dir, os.path.basename(fp)), index=False)
            pd.concat([df[~old], new], ignore_index=True).to_csv(fp, index=False)


if __name__ == '__main__':
    main()
