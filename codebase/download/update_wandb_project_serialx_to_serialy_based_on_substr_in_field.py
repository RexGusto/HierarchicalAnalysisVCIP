import re
import argparse
import wandb


def update_serialx_to_serialy_based_on_substr_in_field(project, x, y, substr, field):
    api = wandb.Api()

    runs = api.runs(path=project, filters={'$and': [
        {'config.serial': 32}, {'config.dataset_name': 'cars'},
    ]})

    len(runs)
    print("Number of runs:", len(runs))

    for run in runs:
        if substr in run.config[field]:
            print(f'Current name: {run.name}', run.config[field])
        
        if run.name.startswith("cars_") and not run.name.startswith("cars_baseline_"):
            new_name = run.name.replace("cars_", "cars_baseline_", 1)
        else:
            print(f"[SKIP NAME] {run.name}")
            new_name = run.name

        print(f"[RENAME] {run.name} → {new_name}")
            
        old_value = run.config.get('dataset_name', None)

        print(f"[MATCH] {run.name} | dataset_name={old_value}")

        # update
        run.config['dataset_name'] = 'cars_baseline'
        # run.update()

        print(f"Updated dataset_name=cars_baseline")
        if new_name != run.name:
            run.name = new_name
            run.update()
            print(f"[UPDATED] {new_name}")
        else:
            print("[NO CHANGE]")
        print(f"Changed to run.name")
        
        # run.config['serial'] = y
        # run.update()

        # print(f'Updated run name to {run.name}')


    print(f'Finished changing from {x} to {y} based on {substr} in cfg {field}')

    return 0

def add_hi_prefix_to_model(project):
    api = wandb.Api()

    runs = api.runs(
        path=project,
        filters={
            'config.serial': 32
        }
    )

    print("Number of runs:", len(runs))

    changed_count = 0

    for run in runs:
        old_model = run.config.get('model_name', '')

        # ---- skip if already hi ----
        if old_model.startswith("hi"):
            print(f"[SKIP MODEL] {run.name} ({old_model})")
            continue

        # ---- new model name ----
        new_model = "hi" + old_model

        print(f"[MODEL] {old_model} → {new_model}")

        # ---- update config properly ----
        run.config['model_name'] = new_model

        # ---- update run name ----
        name = run.name

        # assume format: dataset_baseline + model + serial
        try:
            dataset_prefix = name.split("_")[0] + "_baseline_"
            serial_suffix = "_" + name.split("_")[-1]

            # extract middle safely
            model_part = name[len(dataset_prefix):-len(serial_suffix)]

            new_name = dataset_prefix + "hi" + model_part + serial_suffix

        except Exception as e:
            print(f"[ERROR PARSE] {name} | {e}")
            continue

        print(f"[RENAME] {name} → {new_name}")

        if new_name != name:
            run.name = new_name
            # run.update()
            changed_count += 1
            print(f"[UPDATED] {new_name}")
        else:
            print("[NO CHANGE]")

    print(f"Done. Total renamed runs: {changed_count}")

    return 0

def fill_model_name_extractor_serial39(project):
    api = wandb.Api()

    runs = api.runs(
        path=project,
        filters={
            'config.serial': 39,
            'config.dataset_name': {'$ne': 'soylocal'}
        }
    )

    print("Number of runs:", len(runs))

    changed_count = 0
    skipped_count = 0

    for run in runs:
        dataset_name = run.config.get('dataset_name', '')
        extractor = run.config.get('model_name_extractor', '')
        cluster_ratio = run.config.get('n_cluster_ratio', None)

        print(f"[CHECK] {run.name} | dataset={dataset_name} | extractor={extractor} | ratio={cluster_ratio}")

        # ---- skip if BOTH already filled ----
        if extractor and cluster_ratio:
            print("[SKIP] already filled")
            skipped_count += 1
            continue

        new_extractor = None
        new_ratio = None

        # ---- mapping logic ----
        if dataset_name.endswith("_pt_siglipv2"):
            new_extractor = "hivit_base_patch16_siglip_224.v2_webli"
            new_ratio = 50

        elif dataset_name.endswith("_pt_ig1b"):
            new_extractor = "hiresnet50.fb_swsl_ig1b_ft_in1k"
            new_ratio = 70

        else:
            print("[SKIP] dataset not matching pattern")
            skipped_count += 1
            continue

        print(f"[UPDATE] {dataset_name} → extractor={new_extractor}, ratio={new_ratio}")

        # ---- apply updates only if missing ----
        if not extractor:
            run.config['model_name_extractor'] = new_extractor

        if not cluster_ratio:
            run.config['n_cluster_ratio'] = new_ratio

        run.update()

        changed_count += 1

    print(f"Done. Updated: {changed_count}, Skipped: {skipped_count}")

    return 0


def convert_pt_to_pl_dataset(project):
    api = wandb.Api()

    runs = api.runs(
        path=project,
        filters={
            'config.serial': 39,
            'config.dataset_name': {'$ne': 'soylocal'}
        }
    )

    print("Number of runs:", len(runs))

    changed_count = 0
    skipped_count = 0

    for run in runs:
        dataset_name = run.config.get('dataset_name', '')
        name = run.name

        print(f"[CHECK] {name} | dataset={dataset_name}")

        # ---- detect base dataset ----
        base = None
        for d in ["aircraft", "cub", "cars"]:
            if dataset_name.startswith(d + "_"):
                base = d
                break

        if base is None:
            print("[SKIP] unknown dataset")
            skipped_count += 1
            continue

        # ---- match patterns ----
        if dataset_name.endswith("_pt_siglipv2") or dataset_name.endswith("_pt_ig1b"):
            new_dataset = f"{base}_pl"
        else:
            print("[SKIP] not pt dataset")
            skipped_count += 1
            continue

        print(f"[DATASET] {dataset_name} → {new_dataset}")

        # ---- rename run.name (prefix only) ----
        if name.startswith(dataset_name):
            new_name = name.replace(dataset_name, new_dataset, 1)
        else:
            print("[SKIP NAME] prefix mismatch")
            new_name = name

        print(f"[RENAME] {name} → {new_name}")

        # ---- apply updates ----
        run.config['dataset_name'] = new_dataset

        if new_name != name:
            run.name = new_name

        run.update()
        changed_count += 1

        print(f"[UPDATED] {new_name}")

    print(f"Done. Updated: {changed_count}, Skipped: {skipped_count}")

    return 0

import re

def update_soylocal_serial_to_54(project):
    api = wandb.Api()

    runs = api.runs(
        path=project,
        filters={
            'config.serial': {'$in': [32, 39]},
            'config.dataset_name': 'soylocal'
        }
    )

    print("Number of runs:", len(runs))

    changed_count = 0
    skipped_count = 0

    for run in runs:
        old_serial = run.config.get('serial')
        name = run.name

        print(f"[CHECK] {name} | serial={old_serial}")

        # ---- skip if already correct ----
        if old_serial == 54 and name.endswith("_54"):
            print("[SKIP] already updated")
            skipped_count += 1
            continue

        # ---- update config ----
        run.config['serial'] = 54

        # ---- update run name (suffix only) ----
        new_name = re.sub(r'_(32|39)$', '_54', name)

        print(f"[RENAME] {name} → {new_name}")

        if new_name != name:
            run.name = new_name

        # ---- commit ----
        run.update()

        print(f"[UPDATED] serial {old_serial} → 54")

        changed_count += 1

    print(f"Done. Updated: {changed_count}, Skipped: {skipped_count}")

    return 0

def move_soylocal_54_empty_cluster_to_55(project):
    api = wandb.Api()

    runs = api.runs(
        path=project,
        filters={
            'config.dataset_name': 'soylocal',
            'config.serial': 54
        }
    )

    print("Number of runs:", len(runs))

    changed_count = 0
    skipped_count = 0

    for run in runs:
        cluster = run.config.get('n_cluster_ratio', None)
        name = run.name
        old_serial = run.config.get('serial')

        print(f"[CHECK] {name} | serial={old_serial} | cluster={cluster}")

        # ---- skip if already has value ----
        if cluster not in [None, "", 0]:
            print("[SKIP] cluster already set")
            skipped_count += 1
            continue

        # ---- update serial ----
        run.config['serial'] = 55

        # ---- update run name ----
        # replace _54 at end → _55
        new_name = re.sub(r'_54$', '_55', name)

        print(f"[RENAME] {name} → {new_name}")

        if new_name != name:
            run.name = new_name

        # ---- commit ----
        run.update()

        print(f"[UPDATED] serial 54 -> 55")

        changed_count += 1

    print(f"Done. Updated: {changed_count}, Skipped: {skipped_count}")

    return 0

def update_serial32_tgda_false_to_101(project):
    api = wandb.Api()

    runs = api.runs(
        path=project,
        filters={
            'config.serial': 32,
            'config.tgda': False
        }
    )

    print("Number of runs:", len(runs))

    changed_count = 0
    skipped_count = 0

    for run in runs:
        old_serial = run.config.get('serial')
        name = run.name

        print(f"[CHECK] {name} | serial={old_serial}")

        # ---- skip if already correct ----
        if old_serial == 101 and name.endswith("_101"):
            print("[SKIP] already updated")
            skipped_count += 1
            continue

        # ---- update config ----
        run.config['serial'] = 101

        # ---- update run name (suffix only) ----
        new_name = re.sub(r'_32$', '_101', name)

        print(f"[RENAME] {name} → {new_name}")

        if new_name != name:
            run.name = new_name

        # ---- commit ----
        run.update()

        print(f"[UPDATED] serial {old_serial} → 101")

        changed_count += 1

    print(f"Done. Updated: {changed_count}, Skipped: {skipped_count}")

    return 0


def parse_args():
    parser = argparse.ArgumentParser()

    # input
    parser.add_argument('--project_name', type=str, default='nycu_pcs/Hierarchical',
                        help='project_entity/project_name')
    # filters
    parser.add_argument('--serial_x', type=int, default=32)
    parser.add_argument('--serial_y', type=int, default=311)
    parser.add_argument('--substr', type=str, default='resnet3')
    parser.add_argument('--field', type=str, default='model_name')
    args= parser.parse_args()
    return args


def main():
    args = parse_args()

    # update_serialx_to_serialy_based_on_substr_in_field(
    #    args.project_name, args.serial_x, args.serial_y, args.substr, args.field)
    
    # add_hi_prefix_to_model(
    #    args.project_name)
    
    # fill_model_name_extractor_serial39(
    #    args.project_name)
    
    # convert_pt_to_pl_dataset(
    #    args.project_name)
    
    # update_soylocal_serial_to_54(
    #    args.project_name)

    # move_soylocal_54_empty_cluster_to_55(
    #    args.project_name)

    update_serial32_tgda_false_to_101(
        args.project_name)

    return 0


if __name__ == '__main__':
    main()