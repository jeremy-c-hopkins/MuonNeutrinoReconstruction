# Low Energy Muon Neutrino Reconstruction - Convolutional Neural Network
Inspiration and model architecture comes from https://github.com/shiqiyugit/CNN_angular_reco

# Project

This project contains a Convolutional Neural Network (CNN) implementation for reconstructing muon neutrino inelasticity using IceCube simulation data. 

## Description

The primary code file is `cnn_reconstruction.py` which handles the following tasks:

1. Loading and preprocessing the training, validation, and test data from HDF5 files.
2. Building the CNN model architecture.
3. Defining the loss function and optimizer.
4. Training the model using the provided training data.
5. Evaluating the model's performance on the test data.
6. Generating various plots and visualizations to analyze the model's behavior.

The project also includes several utility modules and files, such as:

- `cnn_model.py`: Contains model architecture.
- `model.py`: Provides base and data classes from constructing model. 
- `data_process.py`: Responsible for processing pulse data.
- `plotting.py`: Functions for generating plots and visualizations.

## Requirements

- Python 3.x
- Keras
- TensorFlow
- NumPy
- Scipy
- Matplotlib
- H5py
- Glob

## Usage

1. Ensure you have all the required dependencies installed.
2. Modify the `TrainingArgs` object in the `cnn_reconstruction.py` file to specify the necessary parameters, such as the data directory, output directory, and training settings.
3. Run the `cnn_reconstruction.py` script to train the CNN model and generate the output plots.

## Acknowledgements

This project was developed as part of a research effort in collaboration with the IceCube Neutrino Observatory. The input data and some of the utility functions were provided Dr. Shiqi Yu, IceCube, and other collaborators.

## License

This project is licensed under the [MIT License](LICENSE).
