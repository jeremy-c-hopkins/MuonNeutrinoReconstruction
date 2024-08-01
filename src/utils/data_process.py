import h5py
import pathlib
import sqlite3
import numpy as np
import pandas as pd

from joblib import Memory
from dataclasses import dataclass
from typing import Optional, Tuple

# Caching
cachedir = "./cache"
memory = Memory(cachedir, verbose=0)


# Pads based on the event with the longest length
def zero_pad(group, max_len):
    event_length = len(group)
    padding = max_len - event_length
    if padding > 0:
        padded = group.reindex(
            group.index.tolist()
            + list(range(group.index[-1] + 1, group.index[-1] + 1 + padding))
        )
    else:
        padded = group
    return padded.fillna(0)


# Data class for sublisting pulses
@dataclass
class DOMData:
    dom_x: float
    dom_y: float
    dom_z: float
    charge: float


def get_hdf5(path, header, data) -> pd.DataFrame:

    with h5py.File(path, "r") as hdf:
        data = np.array(hdf.get("labels"))
        header = np.array(hdf.get("output_label_names"))
        header = [str(i).replace("b", "").strip("'") for i in header]

        df = pd.DataFrame(data, columns=header)

    return df


def build_files(directory, header, data) -> pd.DataFrame:

    import pathlib

    hdf5_df = pd.DataFrame()

    pathdir = pathlib.Path(directory)

    for path in pathdir.iterdir():

        print(path)

        if str(path) == "data/archive/.gitignore":
            continue

        with h5py.File(path, "r") as hdf:
            data = np.array(hdf.get("labels"))
            header = np.array(hdf.get("output_label_names"))
            header = [str(i).replace("b", "").strip("'") for i in header]

            df = pd.DataFrame(data, columns=header)

            hdf5_df = pd.concat([hdf5_df, df], ignore_index=True)

    return hdf5_df


def get_db(path) -> Tuple[pd.DataFrame, pd.DataFrame]:

    connect = sqlite3.connect(path)

    truth_cursor = connect.execute("SELECT * FROM truth")
    truth_headers = [description[0] for description in truth_cursor.description]

    pulse_cursor = connect.execute("SELECT * FROM SRTTWOfflinePulsesDC")
    pulse_headers = [description[0] for description in pulse_cursor.description]

    pulse = pd.read_sql("SELECT * FROM SRTTWOfflinePulsesDC", connect)
    truth = pd.read_sql("SELECT * FROM truth", connect)

    pulse_df = pd.DataFrame(pulse, columns=pulse_headers)
    truth_df = pd.DataFrame(truth, columns=truth_headers)

    return pulse_df, truth_df


class PulseDataProcessing:

    def __init__(
        self,
        df_tuple: Tuple[pd.DataFrame],
        dim_pad: Optional[int] = None,
        normalize: Optional[bool] = None,
        norm_method: Optional[str] = "standard_scaler",
    ) -> None:
        """Construct `ProcessData`

        Args:
            df_tuple: Tuple of dataframes, pulse then truth.
            dim_pad: Number of dimensions to keep data after padding.
            normalize: Specify whether to normalize the data.
            norm_method: Normalization method to be used.

        """

        assert all(
            isinstance(df, pd.DataFrame) for df in df_tuple
        ), "pass tuple of pd.Dataframe"
        assert len(df_tuple) == 2, "tuple must have only pulse and truth dataframes"

        self.pulse, self.truth = df_tuple

        if dim_pad is None:
            dim_pad = 3

        assert dim_pad in [2, 3], "dim_pad must be 2 or 3"

        self.dim_pad = dim_pad

        if normalize is None:
            normalize = True

        self.normalize = normalize
        self.padded_state = False

        norm_method_list = ["keras_layers", "standard_scaler", "min_max", "robust"]

        assert norm_method in norm_method_list, "invalid normalization method"
        self.nom_method = norm_method

        self.dom_shape = self._dom_shape()
        self.pulse_shape = tuple(np.flip(np.shape(self.pulse)))

    def _dom_shape(self) -> tuple[float, float]:
        """
        Gets the length of dom_x dom_y and dom_z in space
        """

        positions = ["dom_x", "dom_y", "dom_z"]
        mins = [self.pulse[dom].min(axis=0) for dom in positions]
        maxs = [self.pulse[dom].max(axis=0) for dom in positions]

        return tuple(max(i) - min(i) for i in zip(mins, maxs))

    def clean(self) -> None:
        """
        Cleans pulse and truth data by removing values with inelasticity equal to 1.0. These values
        should not exist as they are not possible.
        """

        mask = self.truth["inelasticity"] != 1.0
        self.truth = self.truth[mask].reset_index(drop=True)
        self.pulse = self.pulse[
            self.pulse.event_no.isin(self.truth.event_no)
        ].reset_index(drop=True)

    def normalize(self) -> None:
        assert self.padded_state is False, "normalize before sublisting"

    def zero_pad(self) -> None:
        """
        Zero pads events to make each event the same length.
        """

        max_len = self.pulse.groupby("event_no").size().max()
        padded = pd.concat(
            [zero_pad(group, max_len) for _, group in self.pulse.groupby("event_no")]
        )

        padded = padded.reset_index(drop=True)

        mask = padded["event_no"] != 0
        padded["event_no"] = padded["event_no"].mask(~mask).ffill().astype(int)

        self.pulse = padded
        self.pulse_shape = tuple(np.flip(np.shape(self.pulse)))
        self.padded_state = True

    def sublist(self) -> None:
        """
        Creates a list of events with each event having a list of data.
        """

        events = [
            [
                [
                    row["dom_x"],
                    row["dom_y"],
                    row["dom_z"],
                    row["dom_time"],
                    row["charge"],
                ]
                for _, row in group.iterrows()
            ]
            for _, group in self.pulse.groupby("event_no")
        ]
        self.pulse = events

    def get_model(self):

        import keras
        from keras import layers

        shape = self.pulse_shape + (1,)

        inputs = keras.Input(shape)

        x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
        x = layers.MaxPool3D(pool_size=2)(x)
        x = layers.BatchNormalization()(x)

        # x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
        # x = layers.MaxPool3D(pool_size=2)(x)
        # x = layers.BatchNormalization()(x)

        # x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
        # x = layers.MaxPool3D(pool_size=2)(x)
        # x = layers.BatchNormalization()(x)

        # x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
        # x = layers.MaxPool3D(pool_size=2)(x)
        # x = layers.BatchNormalization()(x)

        x = layers.GlobalAveragePooling3D()(x)
        x = layers.Dense(units=512, activation="relu")(x)
        x = layers.Dropout(0.3)(x)

        outputs = layers.Dense(units=1, activation="sigmoid")(x)

        # Define the model.
        model = keras.Model(inputs, outputs, name="3dcnn")

        return model


def build_contained_files(directory):

    pathdir = pathlib.Path(directory)

    X_test_DC = None
    X_test_IC = None
    X_train_DC = None
    X_train_IC = None
    X_validate_DC = None
    X_validate_IC = None
    Y_test = None
    Y_train = None
    Y_validate = None

    count = 0

    for path in pathdir.iterdir():

        with h5py.File(path, "r") as hdf:

            if count >= 1:
                X_test_DC = np.concatenate(
                    (np.array(hdf["X_test_DC"]), X_test_DC), axis=0
                )
                X_test_IC = np.concatenate(
                    (np.array(hdf["X_test_IC"]), X_test_IC), axis=0
                )
                X_train_DC = np.concatenate(
                    (np.array(hdf["X_train_DC"]), X_train_DC), axis=0
                )
                X_train_IC = np.concatenate(
                    (np.array(hdf["X_train_IC"]), X_train_IC), axis=0
                )
                X_validate_DC = np.concatenate(
                    (np.array(hdf["X_validate_DC"]), X_validate_DC), axis=0
                )
                X_validate_IC = np.concatenate(
                    (np.array(hdf["X_validate_IC"]), X_validate_IC), axis=0
                )
                Y_test = np.concatenate((np.array(hdf["Y_test"]), Y_test), axis=0)
                Y_train = np.concatenate((np.array(hdf["Y_train"]), Y_train), axis=0)
                Y_validate = np.concatenate(
                    (np.array(hdf["Y_validate"]), Y_validate), axis=0
                )
            else:
                X_test_DC = np.array(hdf["X_test_DC"])
                X_test_IC = np.array(hdf["X_test_IC"])
                X_train_DC = np.array(hdf["X_train_DC"])
                X_train_IC = np.array(hdf["X_train_IC"])
                X_validate_DC = np.array(hdf["X_validate_DC"])
                X_validate_IC = np.array(hdf["X_validate_IC"])
                Y_test = np.array(hdf["Y_test"])
                Y_train = np.array(hdf["Y_train"])
                Y_validate = np.array(hdf["Y_validate"])

            print(len(X_test_DC))
            print(len(X_test_IC))
            print(len(X_train_DC))
            print(len(X_train_IC))
            print(len(Y_test))
            print(len(Y_train))

            count += 1

    f = h5py.File("data/archive/combined.hdf5", "w")

    f.create_dataset("X_test_DC", data=X_test_DC)
    f.create_dataset("X_test_IC", data=X_test_IC)
    f.create_dataset("X_train_DC", data=X_train_DC)

    f.create_dataset("X_train_IC", data=X_train_IC)
    f.create_dataset("X_validate_DC", data=X_validate_DC)
    f.create_dataset("X_validate_IC", data=X_validate_IC)
    f.create_dataset("Y_test", data=Y_test)
    f.create_dataset("Y_train", data=Y_train)
    f.create_dataset("Y_validate", data=Y_validate)
    f.close()


if __name__ == "__main__":

    import time

    start = time.time()
    build_contained_files(
        "/home/bread/MuonNeutrinoReconstruction/src/data/archive/contained"
    )
    end = time.time()
    print(f"Run Time:{end-start:.5f}s")
