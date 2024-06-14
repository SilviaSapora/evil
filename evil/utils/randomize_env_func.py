import os
import numpy as np
import chex
from typing import Any, Tuple, Dict
import xml.etree.ElementTree as et
import json
from etils import epath


def sample_data(attr_data: dict, np_random: np.random.Generator) -> chex.Array:
    min_vals = np.array(attr_data[MIN])
    max_vals = np.array(attr_data[MAX])
    sampled_vals = np_random.uniform(size=len(attr_data[DEFAULT]))
    sampled_vals = (max_vals - min_vals) * sampled_vals + min_vals
    return sampled_vals


CONTINUOUS = "continuous"
DEFAULT = "default"
DISCRETE = "discrete"
JOINT = "joint"
GEOM = "geom"
MIN = "min"
MAX = "max"
OPTION = "option"

VALID_CONTROL_MODE = [
    CONTINUOUS,
    DISCRETE,
    DEFAULT,
]

DEFAULT_ID = 0
DEFAULT_RGB_ARRAY = "rgb_array"
DEFAULT_SIZE = 480


def randomize_env_xml(
    model_path: Any,
    parameter_config_path: Any,
    use_default: bool,
    rng: np.random.RandomState,
) -> Tuple[str, Dict[str, Any]]:
    reference_xml = et.parse(model_path)
    root = reference_xml.getroot()

    modified_attributes = {}
    if not use_default:
        with open(parameter_config_path, "r") as f:
            parameter_config = json.load(f)

        for tag, configs in parameter_config.items():
            modified_attributes.setdefault(tag, {})
            if tag in [OPTION]:
                for attr, attr_data in configs.items():
                    sampled_vals = sample_data(attr_data, rng)
                    root.find(".//{}[@{}]".format(tag, attr)).set(
                        attr, " ".join(map(lambda x: str(x), sampled_vals))
                    )
                    modified_attributes[tag][attr] = sampled_vals
            elif tag in [GEOM, JOINT]:
                for name, attr_dict in configs.items():
                    for attr, attr_data in attr_dict.items():
                        sampled_vals = sample_data(attr_data, rng)
                        root.find(".//{}[@name='{}']".format(tag, name)).set(
                            attr, " ".join(map(lambda x: str(x), sampled_vals))
                        )
                    modified_attributes[tag].setdefault(name, {})
                    modified_attributes[tag][name][attr] = sampled_vals

    xml = et.tostring(root, encoding="unicode", method="xml")
    return xml, modified_attributes


def randomize_env(env_name, use_default, seed):
    rng = np.random.RandomState(seed)
    parameter_config_path = (
        f"/home/silvias/docker/jaxirl/jaxirl/configs/transfer_configs/{env_name}.json"
    )
    xml, modified_attributes = randomize_env_xml(
        os.path.join(
            "/home/silvias/docker/jaxirl/jaxirl/configs/original_xml_configs/",
            f"{env_name}.xml",
        ),
        parameter_config_path,
        use_default,
        rng,
    )
    mujoco_asset_path = epath.resource_path("brax") / f"envs/assets/{env_name}.xml"
    with open(mujoco_asset_path, "w") as f:
        f.write(xml)
