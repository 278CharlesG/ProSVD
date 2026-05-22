# EigenView: SVD-Based Image Feature Extraction & Matching System

EigenView is a lightweight image recognition tool based on **Singular Value Decomposition (SVD)**. It builds a feature space by extracting "principal component basis vectors" from a set of training images (similar to Principal Component Analysis, PCA) and determines if a new image belongs to the same category by calculating its projection residual.

## ✨ Core Features

* **Custom ROI Cropping**: Supports user-defined coordinates to crop Regions of Interest, effectively filtering out background noise.
* **Intelligent Dimensionality Reduction**: Automatically selects the number of singular values needed to retain 95% of the image energy, discarding redundant information.
* **Model Persistence**: Automatically exports the trained model as an `svd_model.npy` file, allowing for future use without retraining.
* **Fast Matching**: Uses projection residuals in the model subspace combined with an adaptive threshold for rapid image classification.

## 🛠️ Environment Dependencies

Before running the script, ensure you have the following libraries installed:

```bash
pip install numpy Pillow
