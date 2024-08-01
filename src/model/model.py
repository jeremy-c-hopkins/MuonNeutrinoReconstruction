import os
import pathlib

from abc import ABC
from datetime import datetime
from dataclasses import dataclass
from typing import Union, Optional


@dataclass
class TrainingArgs:
    activation: str = "linear"
    batch_size: int = 128
    connected_drop_value: float = 0.2
    continue_train: bool = False
    data_dir: str = "src/data/archive/combined.hdf5"
    DC_drop_value: float = 0.2
    epochs: int = 8
    epochs_step_drop: int = 10
    early: bool = False
    IC_drop_value: float = 0.2
    learning_rate: float = 0.001
    load_model_path: str = None
    loss_func: Optional[Union[str, callable]] = None
    lr_drop: float = 0.58
    lr_epoch: int = 64
    lr_func: Optional[callable] = None
    multi_file: bool = False
    network: str = "make_network"
    notify: bool = False
    optimizer: Optional[callable] = None
    oscweight: bool = False
    output_dir: str = "src/data/review"
    save: bool = True
    show: bool = True
    start_epoch: int = 0
    title: str = "Low Energy Muon Neutrino Inelasticity Reconstruction"
    train_variable: str = "inelasticity"
    verbose: int = 2
    zmax: float = 3.14
    zmin: float = 0.0


class Model(ABC):

    def __init__(self, args: TrainingArgs) -> None:
        self.args = args

    def _make_output_dir(self):
        """
        Class method used  for building output directory in location designated
        by TrainingArgs class attribute `output_dir`. Begins by making or selecting
        directory with current date, then makes new directory with incrementing
        directory index.

        :returns: None
        """

        directory = self.args.output_dir

        pathdir = pathlib.Path(directory)
        labels = []

        for path in pathdir.iterdir():
            labels.append(str(path).split("/")[-1])

        today = str(datetime.today().strftime("%Y_%m_%d"))

        if today not in labels:
            path = os.path.join(directory, today)

            os.mkdir(path)

        directory = os.path.join(directory, today)

        pathdir = pathlib.Path(directory)
        labels = []

        if len(os.listdir(directory)) == 0:
            new_dir = str(1)

            self.path = os.path.join(directory, new_dir)

            os.mkdir(self.path)
        else:
            for path in pathdir.iterdir():
                labels.append(int(str(path).split("/")[-1]))

            new_dir = f"{max(labels)+1}"

            self.path = os.path.join(directory, new_dir)

            os.mkdir(self.path)
