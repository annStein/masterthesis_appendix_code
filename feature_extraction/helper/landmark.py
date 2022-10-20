import numpy as np
import pandas as pd


def get_landmark(landmark_data, landmark_name):
    lm_data = pd.DataFrame()
    lm_data['x'] = landmark_data[f"{landmark_name}_x"]
    lm_data['y'] = landmark_data[f"{landmark_name}_y"]
    lm_data['z'] = landmark_data[f"{landmark_name}_z"]
    return lm_data


def compute_center_lr_joints(landmark_data, joint_str):
    """
    computes center of joints which appear left and right (not possible to compute e.g. center between head and feet)
    """
    df = pd.DataFrame({})
    df["x"] = (landmark_data[joint_str + "_right_x"] + landmark_data[
        joint_str + "_left_x"]) / 2
    df["y"] = (landmark_data[joint_str + "_right_y"] + landmark_data[
        joint_str + "_left_y"]) / 2
    df["z"] = (landmark_data[joint_str + "_right_z"] + landmark_data[
        joint_str + "_left_z"]) / 2
    return df


def _entropy_(data):
    p_data = data.value_counts() / len(data)  # probabilities
    return -np.sum(p_data * np.log2(p_data.astype(float)))