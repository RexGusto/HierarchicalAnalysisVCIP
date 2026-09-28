import os
import argparse
import pandas as pd

def extract_dataset_key(name):
    name = str(name).lower()

    if "aircraft" in name:
        return "aircraft"
    if "cars" in name:
        return "cars"
    if "cub" in name:
        return "cub"
    if "soylocal" in name:
        return "soylocal"
    if name == "all":
        return "all"

    return name

def normalize_method_name(m):
    if m.startswith("hi"):
        return m
    return "hi" + m

def normalize_method_names(df):
    df = df.copy()
    df["method"] = df["method"].apply(normalize_method_name)
    return df


def sort_by_serials(df, serial_order):
    df = df.copy()
    cat = pd.Categorical(df["serial"], categories=serial_order, ordered=True)
    df["_serial_order"] = cat

    if "n_cluster_ratio" in df.columns:
        df = df.sort_values(by=["_serial_order", "n_cluster_ratio"], na_position="first")
    else:
        df = df.sort_values(by=["_serial_order"])

    df = df.drop(columns=["_serial_order"])
    return df


def compute_differences(df, baseline_serial):
    rows = []

    group_keys = ["method", "dataset_key"]

    for (method, dataset), sub in df.groupby(group_keys):
        base = sub[sub["serial"] == baseline_serial]
        if base.empty:
            print(f"[WARN] No baseline {baseline_serial} for {method} / {dataset}, skipping.")
            continue

        base_row = base.iloc[0]

        base_ada = float(base_row["ada_ratio"])
        base_cva = float(base_row["cva"])
        base_acc_max = float(base_row["acc_max"])
        base_acc_mean = float(base_row["acc_mean"])
        base_acc_std = float(base_row["acc_std"])

        for _, r in sub.iterrows():
            r = r.copy()

            ada = float(r["ada_ratio"])
            cva = float(r["cva"])
            acc_max = float(r["acc_max"])
            acc_mean = float(r["acc_mean"])
            acc_std = float(r["acc_std"])

            if r["serial"] == baseline_serial:
                # ADA
                r["abs_dif_ada"] = 0.0
                r["rel_dif_ada"] = 0.0

                r["abs_dif_acc_max"] = 0.0
                r["rel_dif_acc_max"] = 0.0

                # ACC MEAN
                r["abs_dif_acc_mean"] = 0.0
                r["rel_dif_acc_mean"] = 0.0

                # ACC STD
                r["abs_dif_acc_std"] = 0.0
                r["rel_dif_acc_std"] = 0.0

                # CVA
                r["abs_dif_cva"] = 0.0
                r["rel_dif_cva"] = 0.0

            else:
                # ---- ADA ----
                r["abs_dif_ada"] = ada - base_ada
                r["rel_dif_ada"] = (
                    100.0 * (ada - base_ada) / base_ada
                    if base_ada != 0 else 0.0
                )

                r["abs_dif_acc_max"] = acc_max - base_acc_max
                r["rel_dif_acc_max"] = (
                    100.0 * (acc_max - base_acc_max) / base_acc_max
                    if base_acc_max != 0 else 0.0
                )

                # ---- ACC MEAN ----
                r["abs_dif_acc_mean"] = acc_mean - base_acc_mean
                r["rel_dif_acc_mean"] = (
                    100.0 * (acc_mean - base_acc_mean) / base_acc_mean
                    if base_acc_mean != 0 else 0.0
                )

                # ---- ACC STD ----
                r["abs_dif_acc_std"] = acc_std - base_acc_std
                r["rel_dif_acc_std"] = (
                    100.0 * (acc_std - base_acc_std) / base_acc_std
                    if base_acc_std != 0 else 0.0
                )

                # ---- CVA ----
                r["abs_dif_cva"] = cva - base_cva
                r["rel_dif_cva"] = (
                    100.0 * (cva - base_cva) / base_cva
                    if base_cva != 0 else 0.0
                )

            rows.append(r)

    return pd.DataFrame(rows)

def build_master_table(df, main_serials):
    # Keep only target serials
    df["dataset_key"] = df["dataset_name"].apply(extract_dataset_key)

    # Exclude aggregated "all" dataset
    df = df[df["dataset_key"] != "all"].copy()

    # Check ADA
    if "ada_ratio" not in df.columns:
        raise ValueError("Input CSV does not contain ada_ratio column!")

    # Ensure required columns exist
    required_cols = ["model_name_extractor", "extractor_layer"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Input CSV does not contain {col} column!")

    # Normalize method names
    df = normalize_method_names(df)

    # Compute differences
    baseline_serial = main_serials[0]
    df = compute_differences(df, baseline_serial)

    # Sort rows
    df = sort_by_serials(df, main_serials)

    # Keep only useful columns
    keep_cols = [
        "serial",
        "dataset_name",
        "method",
        "model_name_extractor",
        "extractor_layer",
        "n_cluster_ratio",
        "acc_max",
        "abs_dif_acc_max",
        "rel_dif_acc_max",
        "lr_acc_max",
        "acc_mean",
        "abs_dif_acc_mean",
        "rel_dif_acc_mean",
        "acc_std",
        "abs_dif_acc_std",
        "rel_dif_acc_std",
        "ada_ratio",
        "abs_dif_ada",
        "rel_dif_ada",
        'cva',
        "abs_dif_cva",
        "rel_dif_cva",

    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols]

    print(df["lr_acc_max"])


    return df


def split_and_save_tables(df, output_dir):
    df = df.copy()

    # Keep only target serials
    df["dataset_key"] = df["dataset_name"].apply(extract_dataset_key)

    # Exclude aggregated "all" dataset
    df = df[df["dataset_key"] != "all"].copy()

    datasets = sorted(df["dataset_key"].unique())

    for dataset in datasets:
        ds_dir = os.path.join(output_dir, dataset)
        os.makedirs(ds_dir, exist_ok=True)

        sub_ds = df[df["dataset_key"] == dataset]

        master_fp = os.path.join(ds_dir, f"summary_pseudo_all_{dataset}.csv")
        sub_ds_to_save = sub_ds.drop(columns="dataset_key")  # drop internal column
        sub_ds_to_save.to_csv(master_fp, index=False)
        print(f"[OK] Saved dataset master table: {master_fp}")

        methods = sorted(sub_ds["method"].unique())

        for method in methods:
            sub = sub_ds[sub_ds["method"] == method]
            if sub.empty:
                continue

            # remove leading "hi" for filename only
            clean_method = method[2:] if method.startswith("hi") else method

            fn = f"summary_pseudo_{clean_method}_{dataset}.csv"
            fp = os.path.join(ds_dir, fn)

            sub_to_save = sub.drop(columns="dataset_key")  # drop internal column
            sub_to_save.to_csv(fp, index=False)
            print(f"[OK] Saved per-method table: {fp}")



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", type=str, required=True, 
                        help="CSV produced by summarize_acc.py (aggregated)")

    parser.add_argument("--main_serials", nargs="+", type=int, default=[32, 23, 39], 
                        help="Serial order: first one is baseline (e.g., 32 23 39)")
    
    parser.add_argument("--output_dir", type=str, default="results_all/ada_pseudo_tables", 
                        help="Where to save outputs")

    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load input
    df = pd.read_csv(args.input_csv)

    # Build master table
    master_df = build_master_table(df, args.main_serials)

    # Save master CSV
    master_fp = os.path.join(args.output_dir, "summary_pseudo_all.csv")
    master_df.to_csv(master_fp, index=False)
    print(f"[OK] Saved master table: {master_fp}")

    # Split per (model, dataset)
    split_and_save_tables(master_df, args.output_dir)


if __name__ == "__main__":
    main()
