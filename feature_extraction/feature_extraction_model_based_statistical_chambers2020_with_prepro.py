import os

import numpy as np
import pandas as pd

from feature_extraction_modules.feature_extraction_base import FeatureExtractionModule
from helper.landmark import _entropy_
from models.input_data import InputData


class FeatureExtractionModelBasedStatistical_Chambers2020_WithPrepro(FeatureExtractionModule):
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

                    # ============================ POSITION FEATURES ============================= #
                    # ATTENTION!!! position requires normed data!!!
                    current_chunk_results_dict.update(
                        cls._get_position_features(landmark_data, "Wrist"))
                    current_chunk_results_dict.update(
                        cls._get_position_features(landmark_data, "Ankle"))

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
            output_filepath_npy = os.path.join(output_target, basename + '_moto_modelbased_features_chambers2020_with_prepro.npy')
            output_filepath_csv = os.path.join(output_target, basename + '_moto_modelbased_features_chambers2020_with_prepro.csv')
            np.save(output_filepath_npy, cls.results_array[landmark_filename])
            cls.results_dfs[landmark_filename].to_csv(output_filepath_csv, index=False)

    @classmethod
    def _get_position_features(cls,
                               landmark_data,
                               dst_landmark
                               ) -> dict:
        current_chunk_results_dict = {}
        # ------------------ absolute position ------------------ #
        right_median_x = np.median(landmark_data.landmarks[f"{dst_landmark}_right_x"])
        right_median_z = np.median(landmark_data.landmarks[f"{dst_landmark}_right_z"])
        right_median_y = np.median(landmark_data.landmarks[f"{dst_landmark}_right_y"])
        left_median_x = np.median(landmark_data.landmarks[f"{dst_landmark}_left_x"])
        left_median_y = np.median(landmark_data.landmarks[f"{dst_landmark}_left_y"])
        left_median_z = np.median(landmark_data.landmarks[f"{dst_landmark}_left_z"])

        median_x = np.mean([right_median_x, left_median_x])
        median_y = np.mean([right_median_y, left_median_y])
        median_z = np.mean([right_median_z, left_median_z])

        # -----
        idf = pd.DataFrame()
        idf['R'] = landmark_data.landmarks[f"{dst_landmark}_right_x"]
        idf['L'] = landmark_data.landmarks[f"{dst_landmark}_left_x"]
        lr_crosscorr_x = idf.corr().loc['L', 'R']

        # -----
        right_iqr_x = np.subtract(*np.percentile(landmark_data.landmarks[f"{dst_landmark}_right_x"], [75, 25]))
        right_iqr_y = np.subtract(*np.percentile(landmark_data.landmarks[f"{dst_landmark}_right_y"], [75, 25]))
        right_iqr_z = np.subtract(*np.percentile(landmark_data.landmarks[f"{dst_landmark}_right_z"], [75, 25]))
        left_iqr_x = np.subtract(*np.percentile(landmark_data.landmarks[f"{dst_landmark}_left_x"], [75, 25]))
        left_iqr_y = np.subtract(*np.percentile(landmark_data.landmarks[f"{dst_landmark}_left_y"], [75, 25]))
        left_iqr_z = np.subtract(*np.percentile(landmark_data.landmarks[f"{dst_landmark}_left_z"], [75, 25]))

        iqr_x = np.mean([right_iqr_x, left_iqr_x])
        iqr_y = np.mean([right_iqr_y, left_iqr_y])
        iqr_z = np.mean([right_iqr_z, left_iqr_z])

        # -----
        right_entropy_x = _entropy_(landmark_data.landmarks[f"{dst_landmark}_right_x"])
        right_entropy_y = _entropy_(landmark_data.landmarks[f"{dst_landmark}_right_y"])
        right_entropy_z = _entropy_(landmark_data.landmarks[f"{dst_landmark}_right_z"])
        left_entropy_x = _entropy_(landmark_data.landmarks[f"{dst_landmark}_left_x"])
        left_entropy_y = _entropy_(landmark_data.landmarks[f"{dst_landmark}_left_y"])
        left_entropy_z = _entropy_(landmark_data.landmarks[f"{dst_landmark}_left_z"])

        right_2d_mean_entropy = np.mean(
            [right_entropy_x, right_entropy_y])
        left_2d_mean_entropy = np.mean([left_entropy_x, left_entropy_y])
        mean_entropy_2d = np.mean([right_2d_mean_entropy, left_2d_mean_entropy])

        right_3d_mean_entropy = np.mean(
            [right_entropy_x, right_entropy_y, right_entropy_z])
        left_3d_mean_entropy = np.mean([left_entropy_x, left_entropy_y, left_entropy_z])
        mean_entropy_3d = np.mean([right_3d_mean_entropy, left_3d_mean_entropy])

        current_chunk_results_dict[f"{dst_landmark}_median_x"] = [median_x]
        current_chunk_results_dict[f"{dst_landmark}_median_y"] = [median_y]
        current_chunk_results_dict[f"{dst_landmark}_median_z"] = [median_z]
        current_chunk_results_dict[f"{dst_landmark}_left_right_crosscorrel_x"] = [lr_crosscorr_x]
        current_chunk_results_dict[f"{dst_landmark}_iqr_x"] = [iqr_x]
        current_chunk_results_dict[f"{dst_landmark}_iqr_y"] = [iqr_y]
        current_chunk_results_dict[f"{dst_landmark}_iqr_z"] = [iqr_z]
        current_chunk_results_dict[f"{dst_landmark}_mean_2d_entropy"] = [mean_entropy_2d]
        current_chunk_results_dict[f"{dst_landmark}_mean_3d_entropy"] = [mean_entropy_3d]

        return current_chunk_results_dict
