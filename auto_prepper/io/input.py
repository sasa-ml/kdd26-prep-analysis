import os

import numpy as np
import pandas as pd
import polars as pl
from PIL import Image

from auto_prepper.utils.exceptions import UnsupportedFormatError
from auto_prepper.utils.helpers import numpy_to_pl


def parse_raw_data(source):
    df = None
    match source:
        case _ if isinstance(source, pl.DataFrame):
            df = from_polars(source)
        case _ if isinstance(source, pd.DataFrame):
            df = from_pandas(source)
        case _ if isinstance(source, dict):
            df = from_dict(source)
        case _ if isinstance(source, np.ndarray):
            df = from_numpy(source)
        case _ if isinstance(source, str):
            match source.split('.')[-1]:
                case 'csv':
                    df = from_csv(source)
                case 'parquet':
                    df = from_parquet(source)
                case 'json':
                    df = from_json(source)
        case _:
            raise UnsupportedFormatError()
    return df


def from_polars(df):
    return df


def from_dict(d):
    return pl.from_dict(d)


def from_csv(source):
    return pl.read_csv(source=source, infer_schema_length=None)


def from_parquet(source):
    return pl.read_parquet(source=source)


def from_json(source):
    return pl.read_json(source=source)


def from_pandas(data):
    # converting all pandas categories to str to avoid errors
    for column in data.select_dtypes(include=['category']).columns:
        data[column] = data[column].astype(str)
    return pl.from_pandas(data)


def from_numpy(data):
    return numpy_to_pl(data)


def from_image_dir(dir_path, image_extension='png'):
    # TODO revise flattenning channels
    images = []
    for file_name in os.listdir(dir_path):
        if file_name.endswith(image_extension):
            img_path = os.path.join(dir_path, file_name)
            img = Image.open(img_path)
            pixels = np.array(img.getdata())
            pixels = np.ndarray.flatten(pixels)
            images.append(pixels)
    try:
        images = np.array(images)
    except ValueError as e:
        # in case of different sized images, resizing to max
        if 'The requested array has an inhomogeneous shape' not in str(e):
            raise e
        max_pixel_count = len(max(images, key=len))
        for i in range(len(images)):
            images[i].resize(max_pixel_count)
        images = np.array(images)
    df = numpy_to_pl(images)
    return df


def from_audio_dir(audio_dir):
    # TODO
    pass
