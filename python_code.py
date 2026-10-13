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
def show_mean_face():
    if mean_face is None:
        return None

    image = mean_face.reshape(HEIGHT, WIDTH)
    image = normalize(image)
    return Image.fromarray(image)


# ============================================================
# SHOW FIRST EIGENFACE
# ============================================================

def show_eigenface():
    if eigenfaces is None:
        return None

    image = eigenfaces[:, 0].reshape(HEIGHT, WIDTH)
    image = normalize(image)
    return Image.fromarray(image)


# ============================================================
# NORMALIZE IMAGE FOR DISPLAY
# ============================================================

def normalize(image):
    image = np.asarray(image)

    low = image.min()
    high = image.max()

    if high == low:
        return np.zeros(image.shape, dtype=np.uint8)

    image = (image - low) / (high - low) * 255
    return image.astype(np.uint8)


# ============================================================
# RECONSTRUCT A FACE
# ============================================================

def reconstruct(path):
    if eigenfaces is None:
        return None, "Run PCA first."

    if path is None:
        return None, "Upload/select a face."

    image = read_image(path)
    U = image.reshape(-1)

    # Projection
    W = eigenfaces.T @ (U - mean_face)

    # Reconstruction
    U_approx = eigenfaces @ W + mean_face

    reconstructed = U_approx.reshape(HEIGHT, WIDTH)

    error = np.linalg.norm(U - U_approx)

    return (
        Image.fromarray(normalize(reconstructed)),
        f"Reconstruction error = {error:.2f}\n\n"
        "W = V^T(U - mean)\n"
        "U_approx = V W + mean"
    )


# ============================================================
# FACE RECOGNITION
# ============================================================

def recognize(path):
    if features is None:
        return "Run PCA first."

    if path is None:
        return "Upload a face."

    image = read_image(path)
    U = image.reshape(-1)

    # Project query face
    W_query = eigenfaces.T @ (U - mean_face)

    # Distance from query to every database face
    distances = np.linalg.norm(
        features - W_query.reshape(-1, 1),
        axis=0
    )

    nearest = np.argmin(distances)

    return (
        "FACE RECOGNITION\n\n"
        f"Predicted person ID: {person_ids[nearest]}\n"
        f"Matched image: {os.path.basename(file_paths[nearest])}\n"
        f"Distance: {distances[nearest]:.2f}\n\n"
        "Smallest PCA distance = closest face."
    )

# ============================================================
# CROSS-AGE / SAME-PERSON VERIFICATION
# ============================================================

def verify_faces(path1, path2):
    if eigenfaces is None:
        return "Run PCA first."

    if path1 is None or path2 is None:
        return "Upload both faces."

    image1 = read_image(path1)
    image2 = read_image(path2)

    U1 = image1.reshape(-1)
    U2 = image2.reshape(-1)

    # PCA projections
    W1 = eigenfaces.T @ (U1 - mean_face)
    W2 = eigenfaces.T @ (U2 - mean_face)

    # Euclidean distance in PCA space
    distance = np.linalg.norm(W1 - W2)

    # If these files come from CACD, obtain their IDs
    id1 = get_person_id(path1)
    id2 = get_person_id(path2)

    same_actual_person = (id1 == id2)

    if same_actual_person:
        actual = "SAME PERSON"
    else:
        actual = "DIFFERENT PEOPLE"

    # A simple data-based threshold
    threshold = get_threshold()

    if threshold is None:
        decision = "THRESHOLD NOT AVAILABLE"
    elif distance <= threshold:
        decision = "SAME PERSON"
    else:
        decision = "DIFFERENT PEOPLE"

    return (
        "CROSS-AGE FACE VERIFICATION\n\n"
        f"Face 1: {os.path.basename(path1)}\n"
        f"Person ID: {id1}\n\n"
        f"Face 2: {os.path.basename(path2)}\n"
        f"Person ID: {id2}\n\n"
        f"PCA distance = {distance:.2f}\n"
        f"Threshold = {threshold:.2f}\n\n"
        f"Predicted: {decision}\n"
        f"Dataset label: {actual}\n\n"
        "Mathematics:\n"
        "W1 = V^T(U1 - mean)\n"
        "W2 = V^T(U2 - mean)\n"
        "d = ||W1 - W2||"
    )


# ============================================================
# LEARN A SIMPLE THRESHOLD
#
# Same-person pairs should generally have smaller distances.
# Different-person pairs should generally have larger distances.
# ============================================================

def get_threshold():
    if features is None:
        return None

    same = []
    different = []

    n = len(person_ids)

    # Limit number of pairs so it stays fast
    pair_limit = 5000
    count = 0

    for i in range(n):
        for j in range(i + 1, n):

            distance = np.linalg.norm(
                features[:, i] - features[:, j]
            )

            if person_ids[i] == person_ids[j]:
                same.append(distance)
            else:
                different.append(distance)

            count += 1

            if count >= pair_limit:
                break

        if count >= pair_limit:
            break

    if len(same) == 0 or len(different) == 0:
        return None

    # Simple midpoint between average same-person
    # and average different-person distances.
    same_mean = np.mean(same)
    different_mean = np.mean(different)

    return (same_mean + different_mean) / 2
# ============================================================
# CROSS-AGE ACCURACY
# ============================================================

def accuracy():
    if features is None:
        return "Run PCA first."

    threshold = get_threshold()

    if threshold is None:
        return (
            "Not enough repeated person IDs in the loaded data "
            "to calculate cross-age verification."
        )

    same_correct = 0
    same_total = 0

    different_correct = 0
    different_total = 0

    n = len(person_ids)
    count = 0
    pair_limit = 5000

    for i in range(n):
        for j in range(i + 1, n):

            distance = np.linalg.norm(
                features[:, i] - features[:, j]
            )

            actual_same = (
                person_ids[i] == person_ids[j]
            )

            predicted_same = (
                distance <= threshold
            )

            if actual_same:
                same_total += 1
                if predicted_same:
                    same_correct += 1
            else:
                different_total += 1
                if not predicted_same:
                    different_correct += 1

            count += 1

            if count >= pair_limit:
                break

        if count >= pair_limit:
            break

    total = same_total + different_total

    if total == 0:
        return "No pairs available."

    correct = same_correct + different_correct
    total_accuracy = correct / total * 100

    same_accuracy = (
        same_correct / same_total * 100
        if same_total else 0
    )

    different_accuracy = (
        different_correct / different_total * 100
        if different_total else 0
    )

    return (
        "CROSS-AGE RECOGNITION RESULTS\n\n"
        f"Threshold = {threshold:.2f}\n\n"
        f"Same-person accuracy = {same_accuracy:.2f}%\n"
        f"Different-person accuracy = {different_accuracy:.2f}%\n"
        f"Overall accuracy = {total_accuracy:.2f}%\n\n"
        f"Same-person pairs = {same_total}\n"
        f"Different-person pairs = {different_total}"
    )

# ============================================================
# SIMPLE GRADIO GUI
# ============================================================

with gr.Blocks(title="PCA Face Recognition") as app:

    gr.Markdown(
        """
# PCA Face Recognition Across Different Ages

**Project 11 Mathematics + Cross-Age Face Verification**

PCA → Eigenfaces → Projection → Face Recognition →
Same Person at Different Ages
"""
    )

    with gr.Tab("PCA"):

        load = gr.Button(
            "Load Database + Run PCA",
            variant="primary"
        )

        status = gr.Textbox(
            label="PCA Results",
            lines=10
        )

        with gr.Row():
            mean = gr.Image(
                label="Mean Face"
            )

            eigen = gr.Image(
                label="First Eigenface"
            )

        load.click(
            load_database,
            outputs=status
        )

        load.click(
            show_mean_face,
            outputs=mean
        )

        load.click(
            show_eigenface,
            outputs=eigen
        )

    with gr.Tab("Face Recognition"):

        face = gr.Image(
            type="filepath",
            label="Upload Face"
        )

        recognize_button = gr.Button(
            "Recognize Face",
            variant="primary"
        )

        recognition_result = gr.Textbox(
            label="Result",
            lines=8
        )

        recognize_button.click(
            recognize,
            inputs=face,
            outputs=recognition_result
        )

    with gr.Tab("Projection / Reconstruction"):

        reconstruction_face = gr.Image(
            type="filepath",
            label="Upload Face"
        )

        reconstruction_button = gr.Button(
            "Project and Reconstruct"
        )

        reconstructed = gr.Image(
            label="Reconstructed Face"
        )

        reconstruction_result = gr.Textbox(
            label="Mathematics",
            lines=8
        )

        reconstruction_button.click(
            reconstruct,
            inputs=reconstruction_face,
            outputs=[
                reconstructed,
                reconstruction_result
            ]
        )

    with gr.Tab("Different Ages"):

        gr.Markdown(
            """
## Same Person at Different Ages

Upload two CACD images.

The program projects both faces into the same PCA space and
compares their feature vectors.

\[
W_1 = V^T(U_1-\bar U)
\]

\[
W_2 = V^T(U_2-\bar U)
\]

\[
d = ||W_1-W_2||_2
\]

A smaller distance means the PCA representations are more similar.
"""
        )

        with gr.Row():

            face1 = gr.Image(
                type="filepath",
                label="Face 1"
            )

            face2 = gr.Image(
                type="filepath",
                label="Face 2"
            )

        verify = gr.Button(
            "Verify Same Person",
            variant="primary"
        )

        verification = gr.Textbox(
            label="Verification",
            lines=12
        )

        verify.click(
            verify_faces,
            inputs=[
                face1,
                face2
            ],
            outputs=verification
        )

        accuracy_button = gr.Button(
            "Calculate Cross-Age Accuracy"
        )

        accuracy_result = gr.Textbox(
            label="Accuracy",
            lines=8
        )

        accuracy_button.click(
            accuracy,
            outputs=accuracy_result
        )


if __name__ == "__main__":
    app.launch(inbrowser=True)
