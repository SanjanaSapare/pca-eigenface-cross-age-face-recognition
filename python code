import os
import re
import numpy as np
import gradio as gr
from PIL import Image



DATABASE_PATH = r"E:\math\lab11\CACD_VS"

# Keep this small at first. Increase to 500 later if it works.
MAX_IMAGES = 300

# All faces are converted to the same size.
HEIGHT = 80
WIDTH = 80

# Global variables
P = None
mean_face = None
eigenfaces = None
features = None

file_paths = []
person_ids = []



# READ CACD FILE NAMES


def get_person_id(path):
    name = os.path.splitext(os.path.basename(path))[0]
    match = re.match(r"^(\d+)_", name)

    if match:
        return match.group(1)

    return name



def read_image(path):
    image = Image.open(path).convert("L")
    image = image.resize((WIDTH, HEIGHT))
    return np.array(image, dtype=np.float64)



# LOAD DATABASE


def load_database():
    global P, mean_face, eigenfaces, features
    global file_paths, person_ids

    file_paths = []
    person_ids = []

    for root, folders, files in os.walk(DATABASE_PATH):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg")):
                file_paths.append(os.path.join(root, file))

    file_paths.sort()
    file_paths = file_paths[:MAX_IMAGES]

    if len(file_paths) < 2:
        return "ERROR: No JPG images were found."

    images = []
    valid_paths = []

    for path in file_paths:
        try:
            image = read_image(path)
            images.append(image.reshape(-1))
            valid_paths.append(path)
        except Exception:
            pass

    if len(images) < 2:
        return "ERROR: Images could not be read."

    file_paths = valid_paths
    person_ids = [get_person_id(p) for p in file_paths]

   # P MATRIX
    # Each column is one face.
  
    P = np.column_stack(images)

    
    # MEAN FACE
   
    mean_face = np.mean(P, axis=1)

    
    # CENTER THE DATA
   
    A = P - mean_face.reshape(-1, 1)

   
    # PROJECT 11:
    # Calculate A^T A instead of A A^T
    
    small_matrix = A.T @ A

    # --------------------------------------------------------
    # EIGENVALUES AND EIGENVECTORS
    # --------------------------------------------------------
    values, vectors = np.linalg.eigh(small_matrix)

    # Largest eigenvalues first
    order = np.argsort(values)[::-1]
    values = values[order]
    vectors = vectors[:, order]

   
    # CONVERT TO IMAGE-SPACE EIGENVECTORS
    #
    # u_i = A v_i
    
    eigenfaces = A @ vectors

    # Remove zero vectors
    norms = np.linalg.norm(eigenfaces, axis=0)
    valid = norms > 1e-10

    eigenfaces = eigenfaces[:, valid]
    values = values[valid]
    norms = norms[valid]

    # Normalize eigenfaces
    eigenfaces = eigenfaces / norms

   
    # Keep at most N-1 useful components
   
    components = min(len(values), len(file_paths) - 1, 80)

    eigenfaces = eigenfaces[:, :components]
    values = values[:components]

   
    # PROJECT ALL DATABASE FACES
    #
    # W = V^T (P - mean)
    # --------------------------------------------------------
    features = eigenfaces.T @ A

    # --------------------------------------------------------
    # ORTHOGONALITY CHECK
    # --------------------------------------------------------
    check = eigenfaces.T @ eigenfaces
    identity = np.eye(eigenfaces.shape[1])
    orthogonal = np.allclose(check, identity, atol=1e-5)

    return (
        "DATABASE + PCA READY\n\n"
        f"Images: {len(file_paths)}\n"
        f"Different person IDs: {len(set(person_ids))}\n"
        f"P matrix: {P.shape}\n"
        f"A^T A matrix: {small_matrix.shape}\n"
        f"Eigenfaces used: {eigenfaces.shape[1]}\n"
        f"Orthogonal: {orthogonal}\n\n"
        "PCA feature for each face:\n"
        "W = V^T (U - mean face)"
    )


# ============================================================
# SHOW MEAN FACE
# ============================================================
