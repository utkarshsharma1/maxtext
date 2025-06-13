"""
Copyright 2025 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import sys
from typing import Sequence

import yaml

from MaxText.configs.types import MaxTextConfig

def load_config_from_yaml_and_argv(argv: Sequence[str]) -> MaxTextConfig:
    """
    Loads the MaxTextConfig from a YAML file specified in argv[1]
    and applies any command line overrides of the form key=value.

    Args:
        argv: sys.argv-like list of arguments.
            argv[1] should be path to YAML file.
            Subsequent arguments optionally in key=value format as overrides.

    Returns:
        An instance of MaxTextConfig.
    """

    if len(argv) < 2:
        raise ValueError("Please specify config YAML path as first argument")

    config_path = argv[1]

    # Load YAML config file into dictionary
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Parse overrides from argv (key=value)
    for arg in argv[2:]:
        if "=" not in arg:
            continue
        key, val = arg.split("=", 1)
        # Convert val to proper type
        try:
            # Use yaml to parse val to proper python types (int, float, bool, etc)
            parsed_val = yaml.safe_load(val)
        except yaml.YAMLError:
            parsed_val = val
        # Support nested keys separated by dots
        apply_override(config_dict, key.strip(), parsed_val)

    # pydantic config object with applied overrides
    return MaxTextConfig(**config_dict)

def apply_override(config_dict: dict, key: str, value):
    """
    Modifies config_dict in-place to apply override given a "dot" separated key.

    Example:
        key="model.base_emb_dim"
        value=1024

        Will set config_dict['model']['base_emb_dim'] = 1024
    """
    keys = key.split(".")
    d = config_dict
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    d[keys[-1]] = value

initialize = load_config_from_yaml_and_argv

if __name__ == "__main__":
    # Simple CLI demo
    cfg = load_config_from_yaml_and_argv(sys.argv)
    print(cfg.model_dump_json(indent=4))

__all__ = ["initialize", "load_config_from_yaml_and_argv"]
