# Fashion MNIST Classification with TensorFlow

Train a neural network to classify fashion items using the Fashion MNIST dataset with TensorFlow.

![Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist/raw/master/doc/img/fashion-mnist-sprite.png)

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
- [File Descriptions](#file-descriptions)
- [Requirements](#requirements)
- [Author](#author)
- [License](#license)

## Overview

This project demonstrates how to train and evaluate a neural network model to classify fashion items such as T-shirts, trousers, and shoes using the Fashion MNIST dataset. It includes two main scripts: `train_and_save_model.py`, which trains a neural network model and saves it to a file, and `load_and_evaluate_model.py`, which loads the trained model and evaluates its performance on the test dataset.

## Installation

1. Clone this repository to your local machine:
   ```
   git clone https://github.com/your_username/fashion-mnist-classification.git
   ```

2. Navigate to the project directory:
   ```
   cd fashion-mnist-classification
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Train the neural network model and save it:
   ```
   python train_and_save_model.py
   ```

2. Load the trained model and evaluate its performance:
   ```
   python load_and_evaluate_model.py
   ```

## File Descriptions

1. **train_and_save_model.py**: Script to load the Fashion MNIST dataset, preprocess the data, define and train a neural network model, and save the trained model to a file.

2. **load_and_evaluate_model.py**: Script to load the saved model, preprocess the test data, evaluate the model's performance on the test dataset, and display the confusion matrix and classification report.

3. **requirements.txt**: File containing the required Python packages for running the scripts.

4. **fashion_mnist_model.h5**: File containing the saved trained model.

## Requirements

- TensorFlow (>=2.0.0)
- NumPy (>=1.17.0)
- Matplotlib (>=3.0.0)
- Seaborn (>=0.9.0)

## Author

[Your Name/Username]

## License

[License Information]

