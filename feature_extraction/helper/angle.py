import numpy as np

def get_angle_from_2_vectors(v0, v1) -> object:
    len_recording = min(np.shape(v0)[0], np.shape(v1)[0])
    angles = np.zeros([len_recording, 1])
    v0_np = v0.to_numpy()
    v1_np = v1.to_numpy()
    for n in range(0, len_recording):
        dot1 = np.dot(v0_np[n, :], v1_np[n, :])
        norm1 = np.linalg.norm(v0_np[n, :])
        norm2 = np.linalg.norm(v1_np[n, :])
        cosi = dot1 / (norm1 * norm2)
        angles[n] = np.arccos(cosi) * (180 / np.pi)
    return angles.flatten()

def _get_angle_of_3_points_(p1, vertex, p2):
    """
    Computes the angles between shoulder, elbow and wrist for every frame

    Parameters
    ----------
    p1 : nx3 df
    vertex : nx3 df
    p2 : nx3 df

    Returns
    -------
    angles_3d : nx1 numpy array

    """
    len_recording = np.shape(p2)[0]

    vector_MU = (vertex - p2).to_numpy()
    vector_ML = (vertex - p1).to_numpy()

    angles_3d = np.zeros([len_recording, 1])
    for n in range(len_recording):
        dot1 = np.dot(vector_ML[n, :], vector_MU[n, :])
        norm1 = np.linalg.norm(vector_ML[n, :])
        norm2 = np.linalg.norm(vector_MU[n, :])
        cosi = dot1 / (norm1 * norm2)
        angles_3d[n] = np.arccos(cosi) * (180 / np.pi)

    return angles_3d