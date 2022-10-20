import os

import numpy as np
import pandas as pd

from definitions import UPPER_BODY_PARTS, LOWER_BODY_PARTS
from helper.landmark import compute_center_lr_joints, get_landmark
from helper.norm_data import move_points_to_origin, norm_single_lm_by_size
from models.input_data import InputData
from preprocessors.preprocessor_base import Preprocessor


class PreprocessorNormData_Chambers2020(Preprocessor):
    """
    Minimalistic preprocessor for demonstration purposes.
    Rounds all numerical IMU data values to a configured decimal place.
    """

    @classmethod
    def is_precondition_fulfilled(
            cls,
            input_data: InputData,
            **kwargs
    ) -> bool:
        for landmark_filename, landmark_data in input_data.kinect_landmark_data_by_filename.items():
            if not 'Wrist_right_x' in landmark_data.landmarks.columns:
                return False
        return True

    @classmethod
    def apply_preprocessing(
            cls,
            input_data: InputData,
            data_source: str,
            **kwargs
    ) -> InputData:

        for landmark_filename, landmark_data in input_data.kinect_landmark_data_by_filename.items():
            try:
                cls.logger.info(f'Norm {landmark_filename} landmark data by trunk (Chambers 2020)...')

                """
                   uses middle between shoulders to middle between hips to norm landmarks
                   :return: normed landmarks
                   """

                landmarks = landmark_data.landmarks.astype(float)
                upper_reference, lower_reference = cls._calc_torso_points_(landmarks)

                basename = os.path.splitext(landmark_filename)[0]
                filepath = os.path.join(data_source, basename + "_body_axis.csv")
                body_ref = pd.read_csv(filepath)

                ref_dist = body_ref["median_body_axis_3d"][0]

                # compute shoulder and hip angles for rotating
                shoulder_angles = cls._compute_lr_joint_angle_(landmarks, "Shoulder")
                hip_angles = cls._compute_lr_joint_angle_(landmarks, "Hip")

                lm_upper = cls._rotate_landmarks_(landmarks, UPPER_BODY_PARTS, shoulder_angles * -1, upper_reference)
                lm_lower = cls._rotate_landmarks_(landmarks, LOWER_BODY_PARTS, hip_angles * -1, lower_reference)

                lm_upper = cls._norm_landmarks_size_(lm_upper, UPPER_BODY_PARTS, upper_reference, ref_dist)
                lm_lower = cls._norm_landmarks_size_(lm_lower, LOWER_BODY_PARTS, lower_reference, ref_dist,
                                                     is_lower=True)

                normed_landmarks = pd.concat([landmarks.timestamp, lm_upper, lm_lower], axis=1)
                upper_reference, lower_reference = cls._calc_torso_points_(normed_landmarks)
                # move all points -> neck to origin
                lm_upper = move_points_to_origin(lm_upper, UPPER_BODY_PARTS, upper_reference)
                lm_lower = move_points_to_origin(lm_lower, LOWER_BODY_PARTS, upper_reference)

                # concatenate timestamps, upper and lower body
                landmark_data.landmarks = pd.concat(
                    [landmarks.timestamp, lm_upper, lm_lower], axis=1)

                cls.logger.info(f'Norm {landmark_filename} landmark data by trunk (Chambers 2020) complete.')
            except Exception:
                cls.logger.exception(f'Exception during processing of {landmark_filename}')
                continue

        return input_data

    @classmethod
    def _calc_torso_points_(cls, landmark_data):
        # calculate reference points (shoulder and hip center)
        upper_ref = compute_center_lr_joints(landmark_data, "Shoulder").astype(float)
        lower_ref = compute_center_lr_joints(landmark_data, "Hip").astype(float)
        return upper_ref, lower_ref

    @classmethod
    def _compute_lr_joint_angle_(cls, landmark_data, joint_str):
        return np.arctan2((landmark_data[joint_str + "_left_y"] - landmark_data[joint_str + "_right_y"]),
                          (landmark_data[joint_str + "_left_x"] - landmark_data[joint_str + "_right_x"]))

    @classmethod
    def _norm_landmarks_size_(cls, landmark_data, landmark_names, reference_point, reference_distance, is_lower=False):
        landmarks_sized = pd.DataFrame()
        for lm in landmark_names:
            temp_lm = norm_single_lm_by_size(landmark_data, lm, reference_point, reference_distance)
            landmarks_sized[f"{lm}_x"] = temp_lm.x
            if is_lower:
                landmarks_sized[f"{lm}_y"] = temp_lm.y + 1
            else:
                landmarks_sized[f"{lm}_y"] = temp_lm.y
            landmarks_sized[f"{lm}_z"] = temp_lm.z
            landmarks_sized[f"{lm}_confidence"] = landmark_data[f"{lm}_confidence"]
        return landmarks_sized


    @classmethod
    def _rotate_landmarks_(cls, landmark_data, landmark_names, angle_rad, origin):
        landmarks_rotated = pd.DataFrame()
        for lm in landmark_names:
            temp_lm = cls._rotate_single_lm_(landmark_data, lm, angle_rad, origin)
            landmarks_rotated[f"{lm}_x"] = temp_lm.x
            landmarks_rotated[f"{lm}_y"] = temp_lm.y
            landmarks_rotated[f"{lm}_z"] = landmark_data[f"{lm}_z"]
            landmarks_rotated[f"{lm}_confidence"] = landmark_data[f"{lm}_confidence"]
        return landmarks_rotated

    @classmethod
    def _rotate_single_lm_(cls, landmark_data, landmark_name, angle_rad, origin):
        # Rotate values of upper body and norm size
        _lm_bef = get_landmark(landmark_data, landmark_name)
        _lm_rotated = pd.DataFrame()

        theta = angle_rad

        _lm_rotated["x"] = origin.x.values + np.cos(theta) * (_lm_bef.x.values - origin.x.values) - np.sin(theta) * (
                _lm_bef.y.values - origin.y.values)
        _lm_rotated["y"] = origin.y.values + np.sin(theta) * (_lm_bef.x.values - origin.x.values) + np.cos(theta) * (
                _lm_bef.y.values - origin.y.values)
        return _lm_rotated