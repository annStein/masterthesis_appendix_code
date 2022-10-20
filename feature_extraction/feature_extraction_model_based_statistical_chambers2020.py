import os

import numpy as np
import pandas as pd

from feature_extraction_modules.feature_extraction_base import FeatureExtractionModule
from helper.angle import _get_angle_of_3_points_
from helper.landmark import get_landmark, _entropy_
from models.input_data import InputData


class FeatureExtractionModelBasedStatistical_Chambers2020(FeatureExtractionModule):
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

                    # ============================ VELOCITY, ACCELERATION FEATURES ============================= #
                    current_chunk_results_dict.update(
                        cls._get_velocity_acceleration_features(landmark_data, "Wrist"))
                    current_chunk_results_dict.update(
                        cls._get_velocity_acceleration_features(landmark_data, "Ankle"))

                    # ========================================== ANGLE FEATURES ========================================== #
                    angles = cls._calc_angles_(landmark_data.landmarks)
                    current_chunk_results_dict.update(cls._get_angle_features_(angles, "Knee"))
                    current_chunk_results_dict.update(cls._get_angle_features_(angles, "Elbow"))

                    angles_3d = cls._calc_3d_angles_(landmark_data.landmarks)
                    current_chunk_results_dict.update(cls._get_3d_angle_features_(angles_3d, "Knee"))
                    current_chunk_results_dict.update(cls._get_3d_angle_features_(angles_3d, "Elbow"))

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
            output_filepath_npy = os.path.join(output_target, basename + '_moto_modelbased_features_chambers2020.npy')
            output_filepath_csv = os.path.join(output_target, basename + '_moto_modelbased_features_chambers2020.csv')
            np.save(output_filepath_npy, cls.results_array[landmark_filename])
            cls.results_dfs[landmark_filename].to_csv(output_filepath_csv, index=False)

    @classmethod
    def _get_velocity_acceleration_features(cls,
                                            landmark_data,
                                            dst_landmark
                                            ) -> dict:
        current_chunk_results_dict = {}
        # ------------------ velocity ------------------ #
        right_velo_x = np.diff(landmark_data.landmarks[f"{dst_landmark}_right_x"])
        right_velo_y = np.diff(landmark_data.landmarks[f"{dst_landmark}_right_y"])
        right_velo_z = np.diff(landmark_data.landmarks[f"{dst_landmark}_right_z"])
        left_velo_x = np.diff(landmark_data.landmarks[f"{dst_landmark}_left_x"])
        left_velo_y = np.diff(landmark_data.landmarks[f"{dst_landmark}_left_y"])
        left_velo_z = np.diff(landmark_data.landmarks[f"{dst_landmark}_left_z"])

        # -----
        right_median_velo_x = np.median(right_velo_x)
        right_median_velo_y = np.median(right_velo_y)
        right_median_velo_z = np.median(right_velo_z)
        left_median_velo_x = np.median(left_velo_x)
        left_median_velo_y = np.median(left_velo_y)
        left_median_velo_z = np.median(left_velo_z)

        median_velo_x = np.mean([right_median_velo_x, left_median_velo_x])
        median_velo_y = np.mean([right_median_velo_y, left_median_velo_y])
        median_velo_z = np.mean([right_median_velo_z, left_median_velo_z])

        # -----
        right_iqr_velo_x = np.subtract(*np.percentile(right_velo_x, [75, 25]))
        right_iqr_velo_y = np.subtract(*np.percentile(right_velo_y, [75, 25]))
        right_iqr_velo_z = np.subtract(*np.percentile(right_velo_z, [75, 25]))
        left_iqr_velo_x = np.subtract(*np.percentile(left_velo_x, [75, 25]))
        left_iqr_velo_y = np.subtract(*np.percentile(left_velo_y, [75, 25]))
        left_iqr_velo_z = np.subtract(*np.percentile(left_velo_z, [75, 25]))

        iqr_velo_x = np.mean([right_iqr_velo_x, left_iqr_velo_x])
        iqr_velo_y = np.mean([right_iqr_velo_y, left_iqr_velo_y])
        iqr_velo_z = np.mean([right_iqr_velo_z, left_iqr_velo_z])

        current_chunk_results_dict[f"{dst_landmark}_median_velocity_x"] = median_velo_x
        current_chunk_results_dict[f"{dst_landmark}_median_velocity_y"] = median_velo_y
        current_chunk_results_dict[f"{dst_landmark}_median_velocity_z"] = median_velo_z
        current_chunk_results_dict[f"{dst_landmark}_iqr_velocity_x"] = iqr_velo_x
        current_chunk_results_dict[f"{dst_landmark}_iqr_velocity_y"] = iqr_velo_y
        current_chunk_results_dict[f"{dst_landmark}_iqr_velocity_z"] = iqr_velo_z

        # ------------------ acceleration ------------------ #
        right_acc_x = np.diff(right_velo_x)
        right_acc_y = np.diff(right_velo_y)
        right_acc_z = np.diff(right_velo_z)
        left_acc_x = np.diff(left_velo_x)
        left_acc_y = np.diff(left_velo_y)
        left_acc_z = np.diff(left_velo_z)

        # -----
        right_iqr_acc_x = np.subtract(*np.percentile(right_acc_x, [75, 25]))
        right_iqr_acc_y = np.subtract(*np.percentile(right_acc_y, [75, 25]))
        right_iqr_acc_z = np.subtract(*np.percentile(right_acc_z, [75, 25]))
        left_iqr_acc_x = np.subtract(*np.percentile(left_acc_x, [75, 25]))
        left_iqr_acc_y = np.subtract(*np.percentile(left_acc_y, [75, 25]))
        left_iqr_acc_z = np.subtract(*np.percentile(left_acc_z, [75, 25]))

        iqr_acc_x = np.mean([right_iqr_acc_x, left_iqr_acc_x])
        iqr_acc_y = np.mean([right_iqr_acc_y, left_iqr_acc_y])
        iqr_acc_z = np.mean([right_iqr_acc_z, left_iqr_acc_z])

        current_chunk_results_dict[f"{dst_landmark}_iqr_acceleration_x"] = iqr_acc_x
        current_chunk_results_dict[f"{dst_landmark}_iqr_acceleration_y"] = iqr_acc_y
        current_chunk_results_dict[f"{dst_landmark}_iqr_acceleration_z"] = iqr_acc_z

        return current_chunk_results_dict

    @classmethod
    def _get_angle_features_(cls, angles, dst_joint):
        current_chunk_results_dict = {}
        right_mean_2d_angle = np.mean(angles[f"{dst_joint}_right_2d"])
        left_mean_2d_angle = np.mean(angles[f"{dst_joint}_left_2d"])
        mean_2d_angle = np.mean([right_mean_2d_angle, left_mean_2d_angle])

        # -----
        right_stdev_2d_angle = np.std(angles[f"{dst_joint}_right_2d"])
        left_stdev_2d_angle = np.std(angles[f"{dst_joint}_left_2d"])
        mean_stdev_2d_angle = np.mean([right_stdev_2d_angle, left_stdev_2d_angle])

        # -----
        right_entropy_2d_angle = _entropy_(angles[f"{dst_joint}_right_2d"])
        left_entropy_2d_angle = _entropy_(angles[f"{dst_joint}_left_2d"])
        mean_entropy_2d_angle = np.mean([right_entropy_2d_angle, left_entropy_2d_angle])

        # -----
        idf = pd.DataFrame()
        idf['R'] = angles[f"{dst_joint}_right_2d"]
        idf['L'] = angles[f"{dst_joint}_left_2d"]
        lr_crosscorr_2d_angle = idf.corr().loc['L', 'R']

        current_chunk_results_dict[f"{dst_joint}_right_mean_2d_angle"] = right_mean_2d_angle,
        current_chunk_results_dict[f"{dst_joint}_left_mean_2d_angle"] = left_mean_2d_angle,
        current_chunk_results_dict[f"{dst_joint}_mean_2d_angle"] = mean_2d_angle
        current_chunk_results_dict[f"{dst_joint}_right_stdev_2d_angle"] = right_stdev_2d_angle,
        current_chunk_results_dict[f"{dst_joint}_left_stdev_2d_angle"] = left_stdev_2d_angle,
        current_chunk_results_dict[f"{dst_joint}_mean_stdev_2d_angle"] = mean_stdev_2d_angle
        current_chunk_results_dict[f"{dst_joint}_right_entropy_2d_angle"] = right_entropy_2d_angle,
        current_chunk_results_dict[f"{dst_joint}_left_entropy_2d_angle"] = left_entropy_2d_angle,
        current_chunk_results_dict[f"{dst_joint}_mean_entropy_2d_angle"] = mean_entropy_2d_angle
        current_chunk_results_dict[f"{dst_joint}_lr_crosscorr_2d_angle"] = lr_crosscorr_2d_angle

        # ------------------ angle velocity ------------------ #
        right_2d_angle_velo = cls._get_angle_displacement_(angles[f"{dst_joint}_right_2d"])
        left_2d_angle_velo = cls._get_angle_displacement_(angles[f"{dst_joint}_left_2d"])

        # -----
        right_median_2d_angle_velo = np.median(right_2d_angle_velo)
        left_median_2d_angle_velo = np.median(left_2d_angle_velo)
        mean_median_2d_angle_velo = np.mean([right_median_2d_angle_velo, left_median_2d_angle_velo])

        # -----
        right_iqr_2d_angle_velo = np.subtract(*np.percentile(right_2d_angle_velo, [75, 25]))
        left_iqr_2d_angle_velo = np.subtract(*np.percentile(left_2d_angle_velo, [75, 25]))
        mean_iqr_2d_angle_velo = np.mean([right_iqr_2d_angle_velo, left_iqr_2d_angle_velo])

        current_chunk_results_dict[f"{dst_joint}_right_median_2d_angle_velo"] = right_median_2d_angle_velo
        current_chunk_results_dict[f"{dst_joint}_left_median_2d_angle_velo"] = left_median_2d_angle_velo
        current_chunk_results_dict[f"{dst_joint}_mean_median_2d_angle_velo"] = mean_median_2d_angle_velo
        current_chunk_results_dict[f"{dst_joint}_right_iqr_2d_angle_velo"] = right_iqr_2d_angle_velo
        current_chunk_results_dict[f"{dst_joint}_left_iqr_2d_angle_velo"] = left_iqr_2d_angle_velo
        current_chunk_results_dict[f"{dst_joint}_mean_iqr_2d_angle_velo"] = mean_iqr_2d_angle_velo

        # ------------------ angle acceleration ------------------ #
        right_2d_angle_acc = np.diff(right_2d_angle_velo)
        left_2d_angle_acc = np.diff(left_2d_angle_velo)

        right_iqr_2d_angle_acc = np.subtract(*np.percentile(right_2d_angle_acc, [75, 25]))
        left_iqr_2d_angle_acc = np.subtract(*np.percentile(left_2d_angle_acc, [75, 25]))
        mean_iqr_2d_angle_acc = np.mean([right_iqr_2d_angle_acc, left_iqr_2d_angle_acc])

        current_chunk_results_dict[f"{dst_joint}_right_iqr_2d_angle_acc"] = right_iqr_2d_angle_acc
        current_chunk_results_dict[f"{dst_joint}_left_iqr_2d_angle_acc"] = left_iqr_2d_angle_acc
        current_chunk_results_dict[f"{dst_joint}_mean_iqr_2d_angle_acc"] = mean_iqr_2d_angle_acc

        return current_chunk_results_dict

    @classmethod
    def _get_3d_angle_features_(cls, angles, dst_joint):
        current_chunk_results_dict = {}
        right_mean_3d_angle = np.mean(angles[f"{dst_joint}_right_3d"])
        left_mean_3d_angle = np.mean(angles[f"{dst_joint}_left_3d"])
        mean_3d_angle = np.mean([right_mean_3d_angle, left_mean_3d_angle])

        # -----
        right_stdev_3d_angle = np.std(angles[f"{dst_joint}_right_3d"])
        left_stdev_3d_angle = np.std(angles[f"{dst_joint}_left_3d"])
        mean_stdev_3d_angle = np.mean([right_stdev_3d_angle, left_stdev_3d_angle])

        # -----
        right_entropy_3d_angle = _entropy_(angles[f"{dst_joint}_right_3d"])
        left_entropy_3d_angle = _entropy_(angles[f"{dst_joint}_left_3d"])
        mean_entropy_3d_angle = np.mean([right_entropy_3d_angle, left_entropy_3d_angle])

        # -----
        lr_crosscorr_3d_angle = np.correlate(angles[f"{dst_joint}_right_3d"], angles[dst_joint + "_left_3d"],
                                             mode='valid')

        current_chunk_results_dict[f"{dst_joint}_right_mean_3d_angle"] = right_mean_3d_angle,
        current_chunk_results_dict[f"{dst_joint}_left_mean_3d_angle"] = left_mean_3d_angle,
        current_chunk_results_dict[f"{dst_joint}_mean_3d_angle"] = mean_3d_angle
        current_chunk_results_dict[f"{dst_joint}_right_stdev_3d_angle"] = right_stdev_3d_angle,
        current_chunk_results_dict[f"{dst_joint}_left_stdev_3d_angle"] = left_stdev_3d_angle,
        current_chunk_results_dict[f"{dst_joint}_mean_stdev_3d_angle"] = mean_stdev_3d_angle
        current_chunk_results_dict[f"{dst_joint}_right_entropy_3d_angle"] = right_entropy_3d_angle,
        current_chunk_results_dict[f"{dst_joint}_left_entropy_3d_angle"] = left_entropy_3d_angle,
        current_chunk_results_dict[f"{dst_joint}_mean_entropy_3d_angle"] = mean_entropy_3d_angle
        current_chunk_results_dict[f"{dst_joint}_lr_crosscorr_3d_angle"] = lr_crosscorr_3d_angle

        # ------------------ angle velocity ------------------ #
        right_3d_angle_velo = cls._get_angle_displacement_(angles[dst_joint + "_right_3d"])
        left_3d_angle_velo = cls._get_angle_displacement_(angles[dst_joint + "_left_3d"])

        # -----
        right_median_3d_angle_velo = np.median(right_3d_angle_velo)
        left_median_3d_angle_velo = np.median(left_3d_angle_velo)
        mean_median_3d_angle_velo = np.mean([right_median_3d_angle_velo, left_median_3d_angle_velo])

        # -----
        right_iqr_3d_angle_velo = np.subtract(*np.percentile(right_3d_angle_velo, [75, 25]))
        left_iqr_3d_angle_velo = np.subtract(*np.percentile(left_3d_angle_velo, [75, 25]))
        mean_iqr_3d_angle_velo = np.mean([right_iqr_3d_angle_velo, left_iqr_3d_angle_velo])

        current_chunk_results_dict[f"{dst_joint}_right_median_3d_angle_velo"] = right_median_3d_angle_velo
        current_chunk_results_dict[f"{dst_joint}_left_median_3d_angle_velo"] = left_median_3d_angle_velo
        current_chunk_results_dict[f"{dst_joint}_mean_median_3d_angle_velo"] = mean_median_3d_angle_velo
        current_chunk_results_dict[f"{dst_joint}_right_iqr_3d_angle_velo"] = right_iqr_3d_angle_velo
        current_chunk_results_dict[f"{dst_joint}_left_iqr_3d_angle_velo"] = left_iqr_3d_angle_velo
        current_chunk_results_dict[f"{dst_joint}_mean_iqr_3d_angle_velo"] = mean_iqr_3d_angle_velo

        # ------------------ angle acceleration ------------------ #
        right_3d_angle_acc = np.diff(right_3d_angle_velo)
        left_3d_angle_acc = np.diff(left_3d_angle_velo)

        right_iqr_3d_angle_acc = np.subtract(*np.percentile(right_3d_angle_acc, [75, 25]))
        left_iqr_3d_angle_acc = np.subtract(*np.percentile(left_3d_angle_acc, [75, 25]))
        mean_iqr_3d_angle_acc = np.mean([right_iqr_3d_angle_acc, left_iqr_3d_angle_acc])

        current_chunk_results_dict[f"{dst_joint}_right_iqr_3d_angle_acc"] = right_iqr_3d_angle_acc
        current_chunk_results_dict[f"{dst_joint}_left_iqr_3d_angle_acc"] = left_iqr_3d_angle_acc
        current_chunk_results_dict[f"{dst_joint}_mean_iqr_3d_angle_acc"] = mean_iqr_3d_angle_acc

        return current_chunk_results_dict

    @classmethod
    def _calc_angles_(cls, landmark_data):
        # joints for which an angle should be caluclated --> first joint is where the angle is,
        # 2nd and 3rd joint needed for calculation
        joints = [['Shoulder_right', 'Shoulder_left', 'Elbow_right'],
                  ['Elbow_right', 'Shoulder_right', 'Wrist_right'],
                  ['Hip_right', 'Hip_left', 'Knee_right'],
                  ['Knee_right', 'Hip_right', 'Ankle_right'],
                  ['Shoulder_left', 'Shoulder_right', 'Elbow_left'],
                  ['Elbow_left', 'Shoulder_left', 'Wrist_left'],
                  ['Hip_left', 'Hip_right', 'Knee_left'],
                  ['Knee_left', 'Hip_left', 'Ankle_left']]
        angles_deg = pd.DataFrame()

        for joint_with_angle in joints:
            vertex = get_landmark(landmark_data, joint_with_angle[0])[["x", "y"]]
            p1 = get_landmark(landmark_data, joint_with_angle[1])[["x", "y"]]
            p2 = get_landmark(landmark_data, joint_with_angle[2])[["x", "y"]]

            angles_deg[f"{joint_with_angle[0]}_2d"] = pd.Series(_get_angle_of_3_points_(p1, vertex, p2).flatten())

        return angles_deg

    @classmethod
    def _calc_3d_angles_(cls, landmark_data):
        # joints for which an angle should be caluclated --> first joint is vertex,
        # 2nd and 3rd joint needed for calculation
        joints = [['Shoulder_right', 'Shoulder_left', 'Elbow_right'],
                  ['Elbow_right', 'Shoulder_right', 'Wrist_right'],
                  ['Hip_right', 'Hip_left', 'Knee_right'],
                  ['Knee_right', 'Hip_right', 'Ankle_right'],
                  ['Shoulder_left', 'Shoulder_right', 'Elbow_left'],
                  ['Elbow_left', 'Shoulder_left', 'Wrist_left'],
                  ['Hip_left', 'Hip_right', 'Knee_left'],
                  ['Knee_left', 'Hip_left', 'Ankle_left']]
        angles_deg = pd.DataFrame()

        for joint_with_angle in joints:
            vertex = get_landmark(landmark_data, joint_with_angle[0])
            p1 = get_landmark(landmark_data, joint_with_angle[1])
            p2 = get_landmark(landmark_data, joint_with_angle[2])

            angles_deg[f"{joint_with_angle[0]}_3d"] = pd.Series(_get_angle_of_3_points_(p1, vertex, p2).flatten())

        return angles_deg

    # @classmethod
    # def _get_2d_angle_of_3_points_(cls, p0: pd.DataFrame, p1: pd.DataFrame, p2: pd.DataFrame):
    #     # to understand angle calculation: https://stackoverflow.com/questions/3486172/angle-between-3-points
    #     dot = ((p1.x - p0.x) * (p2.x - p0.x) + (p1.y - p0.y) * (p2.y - p0.y)).astype(float)
    #     det = ((p1.x - p0.x) * (p2.y - p0.y) - (p1.y - p0.y) * (p2.x - p0.x)).astype(float)
    #     return np.arctan2(det, dot) * 180 / np.pi + 0.5

    @classmethod
    def _get_smallest_angle_difference_(cls, x, y):
        smallest_angle = np.min(np.abs([y - x, y - x + 360, y - x - 360]), axis=0)
        return smallest_angle

    @classmethod
    def _get_angle_displacement_(cls, angles_of_one_joint):  # different from other deltas - need shortest path
        return cls._get_smallest_angle_difference_(angles_of_one_joint[0:len(angles_of_one_joint) - 1].reset_index(drop=True),
                                        angles_of_one_joint[1:len(angles_of_one_joint)].reset_index(drop=True))
