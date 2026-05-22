import numpy as np
from PIL import Image
import os

# Get current working directory
dir_path = os.getcwd()

# User input for ROI (Region of Interest) coordinates: x1, y1, x2, y2
print("Please enter ROI coordinates (x1 y1 x2 y2, e.g., 10 10 90 90):")
x1, y1, x2, y2 = map(int, input().split())

# Loop to collect training PNG file names
training_files = []
while True:
    print("Please enter training PNG filename (e.g., train1.png; enter 'done' to finish):")
    fname = input().strip()
    if fname.lower() == 'done':
        break
    if not fname.endswith('.png'):
        print("Must be a PNG file.")
        continue
    full_path = os.path.join(dir_path, fname)
    if not os.path.exists(full_path):
        print("File does not exist.")
        continue
    training_files.append(full_path)

# Load and process training images: convert to grayscale, crop ROI, flatten to vector
train_vectors = []
for f in training_files:
    img = Image.open(f).convert('L')  # Convert to grayscale
    crop = img.crop((x1, y1, x2, y2))  # Crop specific pixel area
    mat = np.array(crop)
    vec = mat.flatten()  # Flatten to 1D vector
    train_vectors.append(vec)

# If training data exists, build matrix A (pixels as rows, images as columns), perform SVD
if train_vectors:
    A = np.column_stack(train_vectors)
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    # Automatically select k: retain 95% energy
    cumulative_energy = np.cumsum(S ** 2) / np.sum(S ** 2)
    k = np.argmax(cumulative_energy >= 0.95) + 1
    U_k = U[:, :k]  # Model basis vectors

    # Export model to file for external testing
    np.save('svd_model.npy', U_k)
    print(f"Model trained and exported to 'svd_model.npy', retained {k} singular values.")
else:
    print("No training images, exiting.")
    exit()

# Test loop: user enters test image filename for recognition
while True:
    print("Please enter test PNG filename (e.g., test.png; enter 'quit' to exit):")
    fname = input().strip()
    if fname.lower() == 'quit':
        break
    if not fname.endswith('.png'):
        print("Must be a PNG file.")
        continue
    full_path = os.path.join(dir_path, fname)
    if not os.path.exists(full_path):
        print("File does not exist.")
        continue

    # Load test image, crop ROI, and flatten
    img = Image.open(full_path).convert('L')
    crop = img.crop((x1, y1, x2, y2))
    test_mat = np.array(crop)
    test_vec = test_mat.flatten()

    # Project into subspace, calculate residual
    proj = U_k @ (U_k.T @ test_vec)
    residual = np.linalg.norm(test_vec - proj)

    # Threshold judgment
    thresh = np.mean(S[:k]) * 0.5
    if residual < thresh:
        print(f"Recognition successful! Residual: {residual:.2f} (Below threshold {thresh:.2f})")
    else:
        print(f"Not recognized. Residual: {residual:.2f} (Above threshold {thresh:.2f})")