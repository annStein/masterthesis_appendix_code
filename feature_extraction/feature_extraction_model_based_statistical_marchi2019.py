import os

import numpy as np
import pandas as pd

from feature_extraction_modules.feature_extraction_base import FeatureExtractionModule
from helper.landmark import get_landmark
from models.input_data import InputData


class FeatureExtractionModelBased_Marchi2019(FeatureExtractionModule):
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
            **kwargs
    ) -> None:
        # Initialisation of data array
        for landmark_filename, landmark_data in input_data.kinect_landmark_data_by_filename.items():
            try:
                if len(landmark_data.landmarks) == 90:
                    current_chunk_results_dict = {}
                    current_chunk_results_dict.update(cls._get_kinematic_features(landmark_data.landmarks, ["Shoulder_left", "Shoulder_right", "Elbow_left",
                                                                          "Elbow_right", "Wrist_left", "Wrist_right"]))

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
            output_filepath_npy = os.path.join(output_target, basename + '_moto_modelbased_features_marchi2019.npy')
            output_filepath_csv = os.path.join(output_target, basename + '_moto_modelbased_features_marchi2019.csv')
            np.save(output_filepath_npy, cls.results_array[landmark_filename])
            cls.results_dfs[landmark_filename].to_csv(output_filepath_csv, index=False)

    @classmethod
    def _get_kinematic_features(cls, landmarks, joints: list) -> dict:
        result = {}
        for joint in joints:
            xyz_joint = get_landmark(landmarks, joint)
            # velocity
            velo = np.diff(xyz_joint, axis=0).astype(float)
            result[f"{joint}_std_velocity_x"] = [np.std(velo[:, 0])]
            result[f"{joint}_std_velocity_y"] = [np.std(velo[:, 1])]
            result[f"{joint}_std_velocity_z"] = [np.std(velo[:, 2])]
            velo_total = np.linalg.norm(velo, axis=1)
            result[f"{joint}_std_velocity"] = [np.std(velo_total)]
            # acceleration
            acc = np.diff(velo, axis=0).astype(float)
            result[f"{joint}_std_acceleration_x"] = [np.std(acc[:, 0])]
            result[f"{joint}_std_acceleration_y"] = [np.std(acc[:, 1])]
            result[f"{joint}_std_acceleration_z"] = [np.std(acc[:, 2])]
            acc_total = np.linalg.norm(acc, axis=1)
            result[f"{joint}_std_acceleration"] = [np.std(acc_total)]
        return result