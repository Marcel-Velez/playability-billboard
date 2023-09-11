import os

from .billboard import BillBoard
from .custom import CustomChordSong

def retrieve(dataset_name, base_dir=None):

    assert dataset_name in ["billboard_salami", "custom"], f"{dataset_name} not available"

    if dataset_name == "billboard_salami":
        dataset = BillBoard
    elif dataset_name == "custom":
        dataset = CustomChordSong

    return dataset




