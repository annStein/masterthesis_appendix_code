import os

import pandas as pd

from definitions import BODY_PARTS
from helper.landmark import compute_center_lr_joints
from helper.norm_data import move_points_to_origin, norm_single_lm_by_size
from models.input_data import InputData
from preprocessors.preprocessor_base import Preprocessor


class PreprocessorNormData_Marchi2019(Preprocessor):
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
                cls.logger.info(f'Norm {landmark_filename} landmark data by shoulder (Marchi 2019)...')

                """
                   uses middle between shoulders as keypoint, norms by distance between shoulders
                   :return: normed landmarks
                   """

                landmarks = landmark_data.landmarks.astype(float)

                basename = os.path.splitext(landmark_filename)[0]
                filepath = os.path.join(data_source, basename + "_shoulder_distance.csv")
                shoulder_ref = pd.read_csv(filepath)

                ref_dist = shoulder_ref["median_shoulder_2d_distance"][0]
                ref_point = compute_center_lr_joints(landmarks, "Shoulder")

                normed_landmarks = cls._norm_landmarks_size_(landmarks, BODY_PARTS, ref_dist, ref_point)

                # recalc ref_point
                ref_point = compute_center_lr_joints(normed_landmarks, "Shoulder")
                # neck (shoulder mid) should be origin
                normed_landmarks = move_points_to_origin(normed_landmarks, BODY_PARTS, ref_point)

                # concatenate timestamps, upper and lower body
                landmark_data.landmarks = pd.concat(
                    [landmarks.timestamp, normed_landmarks], axis=1)

                cls.logger.info(f'Norm {landmark_filename} landmark data by shoulders (Marchi 2019) complete.')
            except Exception:
                cls.logger.exception(f'Exception during processing of {landmark_filename}')
                continue

        return input_data

    @classmethod
    def _norm_landmarks_size_(cls, landmark_data, body_parts, reference_distance, reference_point):
        landmarks_sized = pd.DataFrame()
        for lm in body_parts:
            temp_lm = norm_single_lm_by_size(landmark_data, lm, reference_point, reference_distance)
            landmarks_sized[f"{lm}_x"] = temp_lm.x
            landmarks_sized[f"{lm}_y"] = temp_lm.y
            landmarks_sized[f"{lm}_z"] = temp_lm.z
            landmarks_sized[f"{lm}_confidence"] = landmark_data[f"{lm}_confidence"]
        return landmarks_sized
