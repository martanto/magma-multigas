import json
import random
from importlib_resources import files


def generate_random_color():
    data = (files("magma_multigas.resources")
            .joinpath('material-colors.json')
            .read_text())

    color_json = json.loads(data)
    color_key = random.choice(list(color_json.keys()))
    random_color = random.choice(list(color_json[color_key].keys()))
    return color_json[color_key][random_color]
