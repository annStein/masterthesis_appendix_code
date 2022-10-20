import os

import numpy as np
import pandas as pd

from feature_extraction_modules.feature_extraction_base import FeatureExtractionModule
from models.input_data import InputData


class PreprocessingExtractionShoulderDistance(FeatureExtractionModule):
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
                current_chunk_results_dict = {}

                landmarks = landmark_data.landmarks.astype(float)
                shoulder_right = pd.concat([landmarks.Shoulder_right_x, landmarks.Shoulder_right_y, landmarks.Shoulder_right_z], axis=1,
                                           keys=['x', 'y', 'z'])
                shoulder_left = pd.concat([landmarks.Shoulder_left_x, landmarks.Shoulder_left_y, landmarks.Shoulder_left_z], axis=1,
                                          keys=['x', 'y', 'z'])
                dist_3d = np.median(np.linalg.norm(shoulder_right - shoulder_left, axis=1))
                dist_2d = np.median(np.linalg.norm(shoulder_right.iloc[:,:2] - shoulder_left.iloc[:,:2], axis=1))

                current_chunk_results_dict["median_shoulder_2d_distance"] = [dist_2d]
                current_chunk_results_dict["median_shoulder_3d_distance"] = [dist_3d]

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
            output_filepath_npy = os.path.join(output_target, basename + '_shoulder_distance.npy')
            output_filepath_csv = os.path.join(output_target, basename + '_shoulder_distance.csv')
            np.save(output_filepath_npy, cls.results_array[landmark_filename])
            cls.results_dfs[landmark_filename].to_csv(output_filepath_csv, index=False)


