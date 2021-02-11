import argparse
import glob, json, os, sys
from typing import Dict, List

import pandas as pd

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="make csv files for GANImg dataset"
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset/img_78/",
        help="path to a dataset dirctory",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./csv",
        help="a directory where csv files will be saved",
    )
    parser.add_argument(
        "--data_num",
        type=int,
        default=300,
        help="path to a dataset dirctory",
    )

    return parser.parse_args()

def main() -> None:
    args = get_arguments()

    data: Dict[str, List[str]] = {
        "image_path": [],
    }

    for idx in range(args.data_num):
        img_path = os.path.join(args.dataset_dir, "img_%s_%s.jpg")
        data["image_path"].append(img_path % (7, idx))
        data["image_path"].append(img_path % (8, idx))

    # list を DataFrame に変換
    df = pd.DataFrame(
        data,
        columns=["image_path"],
    )
    test_df = pd.DataFrame(
        {
            "image_path": make_test_datapath_list(args.dataset_dir),
        },
    )

    # 保存ディレクトリがなければ，作成
    os.makedirs(args.save_dir, exist_ok=True)

    # 保存
    df.to_csv(os.path.join(args.save_dir, "data.csv"), index=None)
    test_df.to_csv(os.path.join(args.save_dir, "test_data.csv"), index=None)

    print("Finished making csv files.")


def make_test_datapath_list(dataset_dir: str):
    test_img_list = list()

    for img_idx in range(5):
        img_path = os.path.join(dataset_dir, "img_7_" + str(img_idx)+".jpg")
        test_img_list.append(img_path)

        img_path = os.path.join(dataset_dir, "img_8_" + str(img_idx)+".jpg")
        test_img_list.append(img_path)

        img_path = os.path.join(dataset_dir, "img_2_" + str(img_idx)+".jpg")
        test_img_list.append(img_path)

    return test_img_list


if __name__ == "__main__":
    main()
