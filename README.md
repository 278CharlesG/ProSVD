# SVD-Based Image Feature Extraction & Matching System

This is a lightweight image recognition script based on **Singular Value Decomposition (SVD)**. It builds a feature space by extracting "principal component basis vectors" from a set of training images (similar to Principal Component Analysis, PCA) and determines if a new image belongs to the same category by calculating its projection residual.

## Core Features

* **Custom ROI Cropping**: Supports user-defined coordinates to automatically crop Regions of Interest for analysis, effectively filtering out background noise.
* **Intelligent Dimensionality Reduction**: Automatically calculates and retains singular values containing 95% of the energy, discarding redundant information.
* **Model Persistence**: Automatically generates and exports an `svd_model.npy` model file after training, facilitating subsequent use.
* **Fast Matching**: Performs rapid classification by calculating the projection residual of the test image in the model subspace, combined with an adaptive threshold.

##  Environment Dependencies

Before running the script, please ensure that you have installed the following dependency libraries in your Python environment:

```bash
pip install numpy Pillow
