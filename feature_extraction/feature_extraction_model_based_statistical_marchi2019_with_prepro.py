import os

import numpy as np
import pandas as pd

from feature_extraction_modules.feature_extraction_base import FeatureExtractionModule
from helper.landmark import get_landmark
from models.input_data import InputData


class FeatureExtractionModelBased_Marchi2019_WithPrepro(FeatureExtractionModule):
    results_array = {}
    results_dfs = {}

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
    def execute(
            cls,
            input_data: InputData,
            output_target: str,
            chunk_index: int,
            chunk_size: int,
            **kwargs
    ) -> None:
        # Initialisation of data array
        for landmark_filename, landmark_data in input_data.kinect_landmark_data_by_filename.items():
            try:
                if len(landmark_data.landmarks) == 90:
                    current_chunk_results_dict = {}
                    current_chunk_results_dict.update(cls._get_distance_between_2_joints_(landmark_data.landmarks, [["Wrist_left", "Wrist_right"]]))
                    current_chunk_results_dict.update(cls._get_mean_locations_of_joints_(landmark_data.landmarks, ["Elbow_left", "Elbow_right",
                                                                                                                   "Wrist_left", "Wrist_right"]))
                    current_chunk_results_dict.update(
                        cls._get_distance_between_2_joints_(landmark_data.landmarks, [["Shoulder_left", "Wrist_left"], ["Shoulder_right", "Wrist_right"]]))
                    current_chunk_results_dict.update(
                        cls._get_spectrum_of_joints_(landmark_data.landmarks, ["Wrist_left", "Wrist_right"]))

                    current_chunk_results_df = pd.DataFrame(current_chunk_results_dict)
                    current_chunk_results = current_chunk_results_df.to_numpy()
                    if landmark_filename not in cls.results_array.keys():
                        cls.results_array[landmark_filename] = current_chunk_results
                        cls.results_dfs[landmark_filename] = current_chunk_results_df
                    else:
                        cls.results_array[landmark_filename] = np.vstack([cls.results_array[landmark_filename],
                                                                          current_chunk_results])
                        cls.results_dfs[landmark_filename] = cls.results_dfs[landmark_filename].append(
                            current_chunk_results_df, ignore_index=True)
            except Exception:
                cls.logger.exception(f'Exception during processing of {landmark_filename}')
                continue

    @classmethod
    def finalize(cls, output_target: str, **kwargs) -> None:
        for landmark_filename in cls.results_array:
            basename = os.path.splitext(landmark_filename)[0]
            output_filepath_npy = os.path.join(output_target, basename + '_moto_modelbased_features_marchi2019_with_prepro.npy')
            output_filepath_csv = os.path.join(output_target, basename + '_moto_modelbased_features_marchi2019_with_prepro.csv')
            np.save(output_filepath_npy, cls.results_array[landmark_filename])
            cls.results_dfs[landmark_filename].to_csv(output_filepath_csv, index=False)

    @classmethod
    def _get_mean_locations_of_joints_(cls, landmarks, joints: list) -> dict:
        result = {}
        for joint in joints:
            result[f"mean_position_{joint}_x"] = [np.mean(landmarks[f"{joint}_x"])]
            result[f"mean_position_{joint}_y"] = [np.mean(landmarks[f"{joint}_y"])]
            result[f"mean_position_{joint}_z"] = [np.mean(landmarks[f"{joint}_z"])]
        return result

    @classmethod
    def _get_distance_between_2_joints_(cls, landmarks, joint_pairs: list) -> dict:
        """ joint_pairs should have shape (n, 2) """
        result = {}
        for joint_pair in joint_pairs:
            first_joint = get_landmark(landmarks, joint_pair[0]).astype(float)
            second_joint = get_landmark(landmarks, joint_pair[1]).astype(float)
            result[f"distance_3d_{joint_pair[0]}_{joint_pair[1]}"] = [np.mean(np.linalg.norm(second_joint - first_joint, axis=1))]
            result[f"distance_2d_{joint_pair[0]}_{joint_pair[1]}"] = [np.mean(np.linalg.norm(second_joint.iloc[:,:2] - first_joint.iloc[:,:2], axis=1))]
        return result

    @classmethod
    def _get_spectrum_of_joints_(cls, landmarks, joint_list, n_max_values=3):
        """
        n_max_values: number of max values which should be used for feature extraction
        """
        result = {}
        for joint in joint_list:
            min_x = np.min(landmarks[f"{joint}_x"])
            max_x = np.max(landmarks[f"{joint}_x"])
            min_y = np.min(landmarks[f"{joint}_y"])
            max_y = np.max(landmarks[f"{joint}_y"])
            min_z = np.min(landmarks[f"{joint}_z"])
            max_z = np.max(landmarks[f"{joint}_z"])

            result[f"{joint}_spectrum_2d"] = (max_x-min_x) * (max_y-min_y)
            result[f"{joint}_spectrum_3d"] = (max_x-min_x) * (max_y-min_y) * (max_z-min_z)
        return result