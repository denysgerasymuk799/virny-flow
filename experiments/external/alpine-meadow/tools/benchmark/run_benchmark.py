import os
import argparse
import subprocess
import time
import re

import pandas as pd


# load datasets
DATASETS = {}
for file in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")):
    if file.startswith("datasets_"):
        csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets", file)
        id_ = file[9:].split('.')[0]
        DATASETS[id_] = pd.read_csv(csv_path)['Dataset'].values

TEST_IDS = [id_ for id_ in DATASETS if id_.startswith("test")]

LARGE_DATASETS = {"1567_poker_hand", "LL0_1217_click_prediction_small", "LL0_1476_gas_drift", "LL0_197_cpu_act",
                  "LL0_308_puma32h", "LL0_562_cpu_small", "LL0_1113_kddcup99", "LL0_1219_click_prediction_small",
                  "LL0_155_pokerhand", "LL0_23397_comet_mc_sample", "LL0_4541_diabetes130us", "LL0_573_cpu_act",
                  "LL0_1122_ap_breast_prostate", "LL0_1220_click_prediction_small", "LL0_1569_poker_hand",
                  "LL0_298_coil2000", "LL0_511_plasma_retinol", "LL0_1176_internet_advertisements",
                  "LL0_1457_amazon_commerce_reviews", "LL0_180_covertype", "LL0_301_ozone_level", "LL0_558_bank32nh",
                  "LL0_300_isolet"}


def parse_args():
    parser = argparse.ArgumentParser(description="Converting traces")
    parser.add_argument("--datasets", type=str, help="datasets path", required=True)
    parser.add_argument("--id", type=str, help="datasets id", required=True)
    parser.add_argument("--exp", type=str, help="experiment type", required=True)
    parser.add_argument("--exp_arguments", type=int, help="experiment arguments", required=True)
    parser.add_argument("--output", type=str, help="output path", required=True)
    parser.add_argument("--use_test_datasets", type=bool, help="use datasets with ids that start with \"test\"",
                        required=False, default=False)
    parser.add_argument("--skip_large_datasets", type=bool, help="skip datasets known to be large", required=False,
                        default=False)
    parser.add_argument("--fe_enabled", type=bool, help="enable feature engineering", required=False, default=False)

    return parser.parse_args()


def run_benchmarks(args):
    datasets_paths = re.split(",", args.datasets)
    for path in datasets_paths:
        assert os.path.exists(path)
    datasets_id = args.id

    print('Running benchmarks!')

    for path in datasets_paths:
        output_dir = os.path.join(args.output, '{}-{}'.format(args.exp, args.exp_arguments))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for sub_dir in os.listdir(path):
            if args.use_test_datasets:
                # Only run on test datasets
                test_id = "test_" + datasets_id
                if test_id not in TEST_IDS or sub_dir not in DATASETS[test_id]:
                    continue
            else:
                if sub_dir.replace('_MIN_METADATA', '') not in DATASETS[datasets_id]:
                    print(f'skip {sub_dir}')
                    continue

            if args.skip_large_datasets:
                # Skip large datasets
                if sub_dir in LARGE_DATASETS:
                    print("Skipping dataset {} due to largeness".format(sub_dir))
                    continue

            # check directories
            dataset = sub_dir
            dataset_dir = os.path.join(path, dataset, dataset + '_dataset')
            problem_dir = os.path.join(path, dataset, dataset + '_problem')
            if not os.path.exists(dataset_dir) or not os.path.exists(problem_dir):
                print('Cannot find directories for dataset {}'.format(dataset))
                continue

            output_path = os.path.join(output_dir, '{}.csv'.format(dataset))
            if os.path.exists(output_path):
                continue

            # start
            print('Running benchmark for {}'.format(dataset), flush=True)
            start = time.time()

            if args.exp == 'dump':
                # subprocess.Popen('python3 exps/run_dump_exp.py {} {} {} {} {}'.format(
                #     dataset_dir, problem_dir, args.exp_arguments, output_path, args.enable_feature_engineering))

                print('python ./tools/benchmark/exps/run_dump_exp.py {} {} {} {}'.format(
                    dataset_dir, problem_dir, args.exp_arguments, output_path))
                subprocess.Popen('python ./tools/benchmark/exps/run_dump_exp.py {} {} {} {}'.format(
                    dataset_dir, problem_dir, args.exp_arguments, output_path))
            else:
                assert False

            duration = int(time.time() - start)
            print('Benchmark on dataset {} took {} seconds'.format(dataset, duration), flush=True)


def main():
    args = parse_args()

    # run benchmark
    run_benchmarks(args)


if __name__ == '__main__':
    main()
