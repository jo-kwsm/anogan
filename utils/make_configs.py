import argparse
import dataclasses
import itertools
import os
import sys

import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


from typing import Any, Dict, List, Tuple

from libs.config import Config


def str2bool(val: str) -> bool:
    if isinstance(val, bool):
        return val
    if val.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif val.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="make configuration yaml files.")
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./result",
        help="path to a directory where you want to make config files and directories.",
    )

    fields = dataclasses.fields(Config)

    for field in fields:
        type_func = str2bool if field.type is bool else field.type

        if isinstance(field.default, dataclasses._MISSING_TYPE):
            parser.add_argument(
                f"--{field.name}",
                type=type_func,
                nargs="*",
                required=True,
            )
        else:
            parser.add_argument(
                f"--{field.name}",
                type=type_func,
                nargs="*",
                default=field.default,
            )

    return parser.parse_args()


def parse_params(
    args_dict: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[str], List[List[Any]]]:

    base_config = {}
    variable_keys = []
    variable_values = []

    for k, v in args_dict.items():
        if isinstance(v, list):
            variable_keys.append(k)
            variable_values.append(v)
        else:
            base_config[k] = v

    return base_config, variable_keys, variable_values


def main() -> None:
    args = get_arguments()

    args_dict = vars(args).copy()
    del args_dict["root_dir"]

    base_config, variable_keys, variable_values = parse_params(args_dict)

    product = itertools.product(*variable_values)

    for values in product:
        config = base_config.copy()
        param_list = []

        for k, v in zip(variable_keys, values):
            config[k] = v
            param_list.append(f"{k}-{v}")

        dir_name = "_".join(param_list)
        dir_path = os.path.join(args.root_dir, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        config_path = os.path.join(dir_path, "config.yaml")

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)

    print("Finished making configuration files.")


if __name__ == "__main__":
    main()
