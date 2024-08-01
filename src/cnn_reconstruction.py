# -*- coding: utf-8 -*-
"""cnn.py version 1.1.2 - July 12, 2024

Includes:
    @dataclass
    class TrainingArgs
        args:
            activation: str = "linear"
            batch_size: int = 128
            connected_drop_value: float = 0.2
            data_dir: str = "src/data/archive/combined.hdf5"
            DC_drop_value: float = 0.2
            epochs: int = 8
            epochs_step_drop: int = 10
            early: bool = False
            IC_drop_value: float = 0.2
            learning_rate: float = 0.001
            loss_func: Optional[Union[str, callable]] = None
            lr_drop: float = 0.58
            lr_epoch: int = 64
            lr_func: Optional[callable] = None
            multi_file: bool = False
            network: str = "make_network"
            optimizer: Optional[callable] = None
            oscweight: bool = False
            output_dir: str = "src/data/review"
            save: bool = True
            show: bool = True
            title: str = "Low Energy Muon Neutrino Inelasticity Reconstruction"
            train_variable: str = "inelasticity"
            verbose: int = 2
            zmax: float = 3.14
            zmin: float = 0.0

    class Convolution:
        args:
            TrainingArgs

        Methods:
            @inherited
            _make_output_dir
                args:
                    self
                returns:
                    None
                description:
                    called when class is instantialized, builds dir with current
                    date and incrementing numerical value in dir specified in
                    TrainingArgs argument `output_dir`

            @protected
            _get_training_data
                args:
                    self
                returns:
                    None
                description:
                    called when class is instantialized, gets training data
                    from dir specified in TrainingArgs argument `data_dir`

            @protected
            _build_model
                args:
                    self
                returns:
                    None
                description:
                    called when class is instantialized, builds cnn model

            @protected
            _build_learning_func
                args:
                    self
                returns:
                    None
                description:
                    called when class is instantialized, compiles learning
                    rate function from TrainingArgs argument `lr_func`

            @protected
            _build_loss_function
                args:
                    self
                returns:
                    None
                description:
                    called when class is instantialized, compiles loss function
                    from TrainingArgs argument `loss_func`

            @protected
            _compile_model_params
                args:
                    self
                returns:
                    None
                description:
                    called when class is instantialized, compiles cnn model

            @callable
            fit_model
                args:
                    self
                returns:
                    self.model.history
                        - Keras generated training loss dictionary. Data can be
                          extracted using `history.history[eval]`
                description:
                    callable method that trains the model build in above protected
                    methods

            @callable
            predicted_model
                args:
                    self
                returns:
                    self.reconstruction
                        - Numpy array of reconstructed values using
                          `self.X_test_DC` and `self.X_test_IC` which
                          are loaded in _get_training_data
                description:
                    must be run after calling `fit_model()`. predicts specified
                    reconstruction variable using model weights from `fit_model()`

            @callable
            plot_model
                args:
                    self
                returns:
                    None
                description:
                    handles prediction validation using `matplotlib.axes.hist` and
                    `matplotlib.axes.hist2d`. handles model loss plots using
                    `matplotlib.axes.plot`. axis labels and title are handled via
                    TrainingArgs arguments `train_variable` and `title`


Convolutional neural network used for reconstructing
low energy muon neutrinos with IceCube Neutrino Observatory.
Data must be processed using utils/i3_to_hdf5.py,
utils/flatten_distribution.py, and utils/apply_containmentcut.py.

Docs for data pre-processing:
https://912hani.notion.site/FLERCNN-Tutorial-for-angular-reco-1bb10eeb33a4466daa9aa32ddbd30192

Convolutional neural network code example:
https://www.tensorflow.org/tutorials/images/cnn

Project cloned from:
https://github.com/shiqiyugit/CNN_angular_reco.git

tensorboard available:
    run in terminal:
        tensorboard --logdir /path/to/logs/
"""
#########################
# Version of CNN on 8 Nov 2019
#
# Runs net and plots
# Takes in multiple files to train on (large dataset)
# Runs Energy and Zenith only (must do both atm)
####################################

import os
import h5py
import time
import glob
import types
import keras
import atexit
import warnings
import tensorflow as tf

from keras import losses
from inspect import signature

# from utils.log import *
from model.cnn_model import *
from utils.plotting import *
from model.model import Model, TrainingArgs

print(f"Tensorflow Version: {tf.__version__}")

print(f"Keras Version: {keras.__version__}")

tf.config.threading.set_inter_op_parallelism_threads(12)
tf.config.threading.set_intra_op_parallelism_threads(12)


# TODO - Working on combining with Model base class
class Convolution(Model):

    def __init__(self, args: TrainingArgs) -> None:

        super().__init__(args)

        self.callbacks = []

        if self.args.save == True:
            if not self.args.continue_train:
                super()._make_output_dir()
            else:
                self.path = self.args.output_dir

        self._get_training_files()

        self._get_training_data()

        self._build_model()

        self._build_learning_func()

        self._build_loss_function()

        self._compile_model_params()

    def __str__(self) -> str:
        if self.built:
            return self.model.summary()
        else:
            return "Convolutional Neural Network"

    def _get_training_files(self):

        if self.args.multi_file:

            assert os.path.isdir(
                self.args.data_dir
            ), "`TrainingArgs.data_dir` must pass directory when `TrainingArgs.multi_file = True`"

            files_with_paths = os.path.join(
                self.args.data_dir,
                "*_contained.hdf5",
            )

            self.files = sorted(glob.glob(files_with_paths))

            if len(self.files) > self.args.epochs:
                warnings.warn(
                    "there are more files than epochs, some files will not be trained on"
                )

            return self.files

        else:

            assert os.path.isfile(
                self.args.data_dir
            ), "`TrainingArgs.data_dir` must pass file when `TrainingArgs.multi_file = True`"

            self.files = self.args.data_dir

            return self.files

    def _get_training_data(self, hdf=None):
        """
        Protected method that gets model training data using TrainingArgs data class argument `data_dir`.

        :returns: None
        """
        if hdf is None:
            if self.args.multi_file:
                hdf = h5py.File(self.files[0], "r")
            else:
                hdf = h5py.File(self.files, "r")

        self.X_train_DC = hdf["X_train_DC"][:]
        self.X_train_IC = hdf["X_train_IC"][:]
        self.Y_train = hdf["Y_train"][:, 14]

        self.X_validate_DC = hdf["X_validate_DC"][:]
        self.X_validate_IC = hdf["X_validate_IC"][:]
        self.Y_validate = hdf["Y_validate"][:, 14]

        self.X_test_DC = hdf["X_test_DC"][:]
        self.X_test_IC = hdf["X_test_IC"][:]
        self.Y_test = hdf["Y_test"][:, 14]

        assert len(self.Y_train) > 0, "Truth training data is empty"
        assert len(self.Y_validate) > 0, "Truth validating data is empty"
        assert len(self.Y_test) > 0, "Truth testing data is empty"

        assert len(self.X_train_DC) > 0, "DC Feature training data is empty"
        assert len(self.X_validate_DC) > 0, "DC Feature validating data is empty"
        assert len(self.X_test_DC) > 0, "DC Feature testing data is empty"

        assert len(self.X_train_IC) > 0, "IC Feature training data is empty"
        assert len(self.X_validate_IC) > 0, "IC Feature validating data is empty"
        assert len(self.X_test_IC) > 0, "IC Feature testing data is empty"

        self.data = True

    def _build_model(self):
        """
        Protected method for taking training data and building model. Takes
        TrainingArgs data class arguments `connected_drop_value` and `activation`.

        :returns: None
        """

        self.model = make_network(
            self.X_train_DC,
            self.X_train_IC,
            1,
            self.args.DC_drop_value,
            self.args.IC_drop_value,
            self.args.connected_drop_value,
            self.args.activation,
        )

        self.built = True

    # TODO - Change so learning rate function works with _compile_model_params (throws error)
    def _build_learning_func(self):
        """
        Protected method for building learning rate scheduler for keras callbacks.
        Uses TrainingArgs data class arguments `lr_func` and appends the learning
        rate scheduler to the self.callbacks instance variable.

        :returns: None
        """

        if self.args.lr_func is not None:
            sig = signature(self.args.lr_func)

            assert (
                len(sig.parameters) == 4
            ), """
Learning rate function `TrainingArgs.lr_func` should take 4 parameters:
- epoch
    - for calculating the learning rate on given epoch
- initial learning rate
    - starting value for learning rate before training
- learning rate drop
    - the rate at which the learning rate drops per epoch
- epochs step drop
    - how many epochs before changing learning rate

"""

            lr_scheduler = keras.callbacks.LearningRateScheduler(
                lambda epoch: self.args.lr_func
            )

            self.callbacks.append(lr_scheduler)

    def _build_loss_function(self):

        if self.args.loss_func is not None:

            if isinstance(self.args.loss_func, types.FunctionType):

                sig = signature(self.args.loss_func)

                assert (
                    len(sig.parameters) == 2
                ), "Loss function must take two parameters, `truth` and `predicted`"

            else:

                loss_functions = [
                    "mae",
                    "mse",
                    "mean_absolute_error",
                    "mean_squared_error",
                ]

                assert (
                    str(self.args.loss_func) in loss_functions
                ), "must use loss function `mae`, `mean_absolute_error`, `mse`, or `mean_squared_error`"

            self.loss = self.args.loss_func

        else:

            self.loss = "mae"

    # TODO - Figure out how learning rate functions work with keras.optimizer
    def _compile_model_params(self):

        if self.args.early:
            self.callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta=1e-20,
                    patience=10,
                    verbose=1,
                    mode="min",
                    restore_best_weights=True,
                )
            )

        if self.args.optimizer is not None:
            self.optimizer = self.args.optimizer
        else:
            self.optimizer = keras.optimizers.Adam(
                learning_rate=self.args.learning_rate
            )

        if self.args.save:
            log_dir = self.path + "/logs/"

            self.callbacks.append(
                tf.keras.callbacks.TensorBoard(log_dir=log_dir, write_images=True)
            )

        if not self.args.multi_file:
            self.model.compile(
                optimizer=self.optimizer,
                loss=self.loss,
                metrics=[self.loss],
            )

    def fit_model(self):

        if self.args.multi_file:
            self._fit_large_model()
        else:
            self._fit_model()

        return

    def _fit_large_model(self):

        self.history = {"loss": [], "mae": [], "val_loss": [], "val_mae": []}

        self.log_dir = self.path + "/logs/"

        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=self.log_dir, write_images=True
        )

        self.model.compile(
            loss="mae",
            optimizer=keras.optimizers.Adam(learning_rate=self.args.learning_rate),
            metrics=["mae"],
        )

        if self.args.continue_train:
            self.model.load_weights(self.args.load_model_path)
        else:
            weight_path = self.path + "/model_while_running.keras"

        for epoch in range(
            self.args.start_epoch, self.args.start_epoch + self.args.epochs
        ):

            start = time.time()

            current_lr = self.args.learning_rate * math.pow(
                self.args.lr_drop,
                math.floor((1 + epoch / 1000) / self.args.epochs_step_drop),
            )

            self.model.optimizer.learning_rate.assign(current_lr)

            file = self.files[epoch % len(self.files)]

            hdf = h5py.File(file, "r")

            X_train_DC = hdf["X_train_DC"][:]
            X_train_IC = hdf["X_train_IC"][:]
            Y_train = hdf["Y_train"][:, 14]
            X_validate_DC = hdf["X_validate_DC"][:]
            X_validate_IC = hdf["X_validate_IC"][:]
            Y_validate = hdf["Y_validate"][:, 14]

            valid_train = np.where(Y_train > 0)
            X_train_DC = X_train_DC[valid_train]
            X_train_IC = X_train_IC[valid_train]
            Y_train = Y_train[valid_train]

            valid_validate = np.where(Y_validate > 0)
            X_validate_DC = X_validate_DC[valid_validate]
            X_validate_IC = X_validate_IC[valid_validate]
            Y_validate = Y_validate[valid_validate]

            # if epoch > 0:
            #     self.model.load_weights(weight_path)

            history = self.model.fit(
                [X_train_DC, X_train_IC],
                Y_train,
                validation_data=(
                    [X_validate_DC, X_validate_IC],
                    Y_validate,
                ),
                batch_size=self.args.batch_size,
                initial_epoch=epoch,
                epochs=epoch + 1,  # increment to next epoch
                callbacks=[
                    keras.callbacks.ModelCheckpoint(
                        "%s/model_while_running.keras" % self.path
                    ),
                    tensorboard_callback,
                ],
                verbose=2,
            )

            hdf.close()

            del (
                X_train_DC,
                X_train_IC,
                Y_train,
                X_validate_DC,
                X_validate_IC,
                Y_validate,
                file,
                hdf,
            )

            mae = history.history["mae"][0]
            val_mae = history.history["val_mae"][0]

            self.history["loss"].append(history.history["loss"][0])
            self.history["mae"].append(mae)
            self.history["val_loss"].append(history.history["val_loss"][0])
            self.history["val_mae"].append(val_mae)

            end = time.time()

            print(
                f"Epoch: {epoch}, Time: {end-start:.2f}s, Error: {mae:.4f}, Val Error: {val_mae:.4f}, Learning Rate: {current_lr}"
            )

        model_path = self.path + "/keras.keras"

        self.model.save(model_path)

        git = os.path.join(self.log_dir, ".gitignore")

        with open(git, "w") as f:
            f.write("*")

        self.fit = True

    def _fit_model(self):

        self.history = self.model.fit(
            [self.X_train_DC, self.X_train_IC],
            self.Y_train,
            validation_data=([self.X_validate_DC, self.X_validate_IC], self.Y_validate),
            batch_size=self.args.batch_size,
            epochs=self.args.epochs,
            callbacks=self.callbacks,
            verbose=self.args.verbose,
        )

        model_path = self.path + "/keras.keras"

        self.model.save(model_path)

        self.fit = True

    def predict_model(self):

        if self.args.multi_file:
            self._predict_large_model()
        else:

            reconstruction = self.model.predict([self.X_test_DC, self.X_test_IC])

            self.reconstruction = [i[0] for i in reconstruction]

            self.truth = self.Y_test

            return self.reconstruction

        self.reconstructed = True

        return

    def _predict_large_model(self):

        self.reconstruction = []
        self.truth = []

        count = 0

        for file in self.files:

            count += 1

            hdf = h5py.File(file, "r")

            X_test_DC = hdf["X_test_DC"][:]
            X_test_IC = hdf["X_test_IC"][:]
            Y_test = hdf["Y_test"][:, 14]

            valid_test = np.where(Y_test > 0)
            X_test_DC = X_test_DC[valid_test]
            X_test_IC = X_test_IC[valid_test]
            Y_test = Y_test[valid_test]

            pred = self.model.predict([X_test_DC, X_test_IC])

            pred = [i[0] for i in pred]

            self.reconstruction = np.concatenate((self.reconstruction, pred))
            self.truth = np.concatenate((self.truth, Y_test))

            hdf.close()

            del (
                X_test_DC,
                X_test_IC,
                Y_test,
                hdf,
            )

            if count > 5:
                break

    def plot_model(self):

        assert self.fit, "must call `fit_model` before plotting"
        assert self.reconstructed, "must call `predicted_model` before plotting"

        if self.args.multi_file:
            history = self.history
        else:
            history = self.history.history

        self.plot_loss_path = plot_loss(
            history,
            self.path,
            self.args.title,
            True,
            self.args.save,
            self.args.show,
        )

        self.plot_2d_path = reconstructed_hist_2d(
            self.truth,
            self.reconstruction,
            self.path,
            self.args.title,
            self.args.train_variable,
            self.args.save,
            self.args.show,
        )

        self.plot_1d_path = reconstructed_hist_1d(
            self.truth,
            self.reconstruction,
            self.path,
            self.args.title,
            self.args.train_variable,
            self.args.save,
            self.args.show,
        )

        self.plot_2d_error_path = plot_2D_prediction(
            self.truth, self.reconstruction, save=True, savefolder=self.path
        )

        if self.args.notify:
            self.attachments = [
                self.plot_loss_path,
                self.plot_2d_path,
                self.plot_1d_path,
                self.plot_2d_error_path,
            ]

        if self.args.notify:
            self._notify_user()


# args = TrainingArgs(
#     data_dir="/home/bread/Documents/MuonNeutrinoReconstruction/src/data/archive/flercnn_IC19cut",
#     output_dir="/home/bread/Documents/MuonNeutrinoReconstruction/src/data/review/2024_07_17/5",
#     title="Inelasticity Testing",
#     epochs=100,
#     save=True,
#     train_variable="inelasticity",
#     batch_size=128,
#     activation="linear",
#     verbose=1,
#     multi_file=True,
#     notify=False,
#     continue_train=True,
#     load_model_path="/home/bread/Documents/MuonNeutrinoReconstruction/src/data/review/2024_07_17/5/keras.keras",
#     start_epoch=100,
# )

args = TrainingArgs(
    data_dir="/home/bread/Documents/MuonNeutrinoReconstruction/src/data/archive/flercnn_IC19cut",
    output_dir="/home/bread/Documents/MuonNeutrinoReconstruction/src/data/review",
    title="Inelasticity Testing",
    epochs=2,
    save=True,
    train_variable="inelasticity",
    batch_size=128,
    activation="linear",
    verbose=1,
    multi_file=True,
    notify=True,
)

model = Convolution(args=args)

model.fit_model()

model.predict_model()

# model.plot_model()


def ZenithLoss(y_truth, y_predicted):
    return losses.mean_absolute_error(y_truth, y_predicted[:, 0])


def EnergyLoss(y_truth, y_predicted):
    return losses.mean_absolute_error(y_truth, y_predicted)


def InelasticityLoss(y_truth, y_predicted):
    return losses.mean_absolute_error(y_truth, y_predicted)
