import os

import numpy as np
import pandas as pd

from masterthesis_appendix_code.constants.definitions import BODY_PARTS
from feature_extraction_modules.feature_extraction_base import FeatureExtractionModule
from helper.angle import _get_angle_of_3_points_, get_angle_from_2_vectors
from helper.landmark import get_landmark
from models.input_data import InputData


class FeatureExtractionModelBasedHistogram_Mccay2019(FeatureExtractionModule):
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
            number_of_bins: int = 8
    ) -> None:
        # Initialisation of data array
        for landmark_filename, landmark_data in input_data.kinect_landmark_data_by_filename.items():
            try:
                if len(landmark_data.landmarks) == 30:
                    landmarks = landmark_data.landmarks.astype(float)
                    current_chunk_results_dict = {}
                    bones = [['Elbow', 'Shoulder'], ['Wrist', 'Elbow'],
                             ['Knee', 'Hip'], ['Ankle', 'Knee']]

                    angle_joints = [['Shoulder', 'Elbow', 'Wrist'], ['Hip', 'Knee', 'Ankle']]

                    body_axis = cls._calculate_body_axis_(landmarks)

                    # angles
                    histogram_2d_angles = cls._calculate_2d_angles_histogram_(landmarks, bones, body_axis, number_of_bins)
                    current_chunk_results_dict.update(histogram_2d_angles)

                    histogram_3d_angles = cls._calculate_3d_angles_histogram_(landmarks, angle_joints, number_of_bins)
                    current_chunk_results_dict.update(histogram_3d_angles)

                    # displacements
                    displacements = cls._get_landmark_2d_displacement_histogram_(landmarks, BODY_PARTS, body_axis)
                    current_chunk_results_dict.update(displacements)

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
            output_filepath_npy = os.path.join(output_target, basename + '_moto_modelbased_features_mccay2019.npy')
            output_filepath_csv = os.path.join(output_target, basename + '_moto_modelbased_features_mccay2019.csv')
            np.save(output_filepath_npy, cls.results_array[landmark_filename])
            cls.results_dfs[landmark_filename].to_csv(output_filepath_csv, index=False)

    @classmethod
    def _calculate_body_axis_(cls, landmarks) -> pd.DataFrame:
        shoulder_mid_x = landmarks.Shoulder_right_x - landmarks.Shoulder_left_x
        shoulder_mid_y = landmarks.Shoulder_right_y - landmarks.Shoulder_left_y
        shoulder_mid_z = landmarks.Shoulder_right_z - landmarks.Shoulder_left_z

        hip_mid_x = landmarks.Hip_right_x - landmarks.Hip_left_x
        hip_mid_y = landmarks.Hip_right_y - landmarks.Hip_left_y
        hip_mid_z = landmarks.Hip_right_z - landmarks.Hip_left_z

        body_axis = {"x": shoulder_mid_x - hip_mid_x, "y": shoulder_mid_y - hip_mid_y, "z": shoulder_mid_z - hip_mid_z}

        return pd.DataFrame(body_axis)

    @classmethod
    def get_bone(cls, bone_name, landmarks, side):
        """
        side is either left or right
        """
        lm = get_landmark(landmarks, f"{bone_name[0]}_{side}")
        parent_lm = get_landmark(landmarks, f"{bone_name[1]}_{side}")
        bone = parent_lm - lm
        return bone

    @classmethod
    def get_angle_landmarks(cls, joint, landmark_data, side):
        """
        @param joint: consists of 3 landmark names; first upper, then middle (vertex), then lower landmark
        @param landmark_data: all landmarks
        @param side: left or right
        """
        assert len(joint) == 3
        upper_joint = get_landmark(landmark_data, f"{joint[0]}_{side}")
        middle_joint = get_landmark(landmark_data, f"{joint[1]}_{side}")
        lower_joint = get_landmark(landmark_data, f"{joint[2]}_{side}")
        return upper_joint, middle_joint, lower_joint

    @classmethod
    def _calculate_2d_angles_histogram_(cls, landmarks, bone_names, body_axis, bins=8) -> pd.DataFrame:
        """
        @param bone_names: list of bones, one bone consists of 2 joints: landmark and parent-landmark,
        e.g. Elbow and Shoulder
        """
        result = {}

        # Joint orientation
        for bone_name in bone_names:
            # left limb
            bone = cls.get_bone(bone_name, landmarks, "left")
            angles_2d_left = get_angle_from_2_vectors(body_axis[["x","y"]], bone[["x","y"]])
            histogr_prefix = f"{bone_name[0]}_{bone_name[1]}_left_2d_angle"
            result.update(cls._build_angle_histogram(histogr_prefix=histogr_prefix, data=angles_2d_left,
                                                     number_of_bins=bins, histogr_range=(0, 360)))

            # right limb
            bone = cls.get_bone(bone_name, landmarks, "right")
            angles_2d_right = get_angle_from_2_vectors(body_axis[["x","y"]], bone[["x","y"]])
            histogr_prefix = f"{bone_name[0]}_{bone_name[1]}_right_2d_angle"
            result.update(cls._build_angle_histogram(histogr_prefix=histogr_prefix, data=angles_2d_right,
                                                     number_of_bins=bins, histogr_range=(0, 360)))

        return pd.DataFrame(result)

    @classmethod
    def _calculate_3d_angles_histogram_(cls, landmarks, angle_joints, bins=8) -> pd.DataFrame:
        """
       @param joint: consists of 3 landmark names; first upper, then middle (vertex), then lower landmark
        """
        result = {}

        # Joint orientation
        for joint in angle_joints:
            # left limb
            upper_joint, middle_joint, lower_joint = cls.get_angle_landmarks(joint, landmarks, "left")
            angles_3d_left = _get_angle_of_3_points_(upper_joint, middle_joint, lower_joint)
            histogr_prefix = f"{joint[1]}_left_3d_angle"
            result.update(cls._build_angle_histogram(histogr_prefix=histogr_prefix, data=angles_3d_left,
                                                     number_of_bins=bins, histogr_range=(0, 360)))

            # right limb
            upper_joint, middle_joint, lower_joint = cls.get_angle_landmarks(joint, landmarks, "right")
            angles_3d_right = _get_angle_of_3_points_(upper_joint, middle_joint, lower_joint)
            histogr_prefix = f"{joint[1]}_right_3d_angle"
            result.update(cls._build_angle_histogram(histogr_prefix=histogr_prefix, data=angles_3d_right,
                                                     number_of_bins=bins, histogr_range=(0, 360)))

        return pd.DataFrame(result)

    @classmethod
    def _get_landmark_2d_displacement_histogram_(cls, landmarks, body_parts, body_axis, frames_diff=5, number_of_bins=8):
        result = {}
        body_axis_vector = body_axis.iloc[::frames_diff, :]
        landmarks_vector = landmarks.iloc[::frames_diff, :]

        # Joint orientation
        for part in body_parts:
            displacement_vector = cls._get_displacement_vector_(landmarks_vector, part)
            angles_xy = get_angle_from_2_vectors(body_axis_vector[["x","y"]], displacement_vector[["x","y"]])
            histogr_prefix = f"{part}_xy_displacement"
            result.update(cls._build_displacement_histogram(histogr_prefix=histogr_prefix, angles=angles_xy,
                                                            displacements=displacement_vector[["x", "y"]],
                                                            number_of_bins=number_of_bins))

            angles_xz = get_angle_from_2_vectors(body_axis_vector[["x","z"]], displacement_vector[["x","z"]])
            histogr_prefix = f"{part}_xz_displacement"
            result.update(cls._build_displacement_histogram(histogr_prefix=histogr_prefix, angles=angles_xz,
                                                            displacements=displacement_vector[["x", "z"]],
                                                            number_of_bins=number_of_bins))

            angles_yz = get_angle_from_2_vectors(body_axis_vector[["y","z"]], displacement_vector[["y","z"]])
            histogr_prefix = f"{part}_yz_displacement"
            result.update(cls._build_displacement_histogram(histogr_prefix=histogr_prefix, angles=angles_yz,
                                                            displacements=displacement_vector[["y", "z"]],
                                                            number_of_bins=number_of_bins))

        return pd.DataFrame(result)

    @classmethod
    def _get_displacement_vector_(cls, landmarks, body_part):
        displacement = pd.DataFrame()
        displacement["x"] = landmarks[f"{body_part}_x"].diff().iloc[1:]
        displacement["y"] = landmarks[f"{body_part}_y"].diff().iloc[1:]
        displacement["z"] = landmarks[f"{body_part}_z"].diff().iloc[1:]
        return displacement

    @classmethod
    def _build_angle_histogram(cls, histogr_prefix, data, number_of_bins=8, histogr_range=(0, 360)) -> dict:
        result = {}
        values, bins = np.histogram(data, number_of_bins, histogr_range)
        for val, _bin in zip(values, bins):
            result[f"{histogr_prefix}_{str(int(_bin))}"] = [val]
        return result

    @classmethod
    def _build_displacement_histogram(cls, histogr_prefix, angles, displacements, number_of_bins=8,
                                      histogr_range=(0, 360)) -> dict:
        result = {}
        step = (histogr_range[1] - histogr_range[0]) // number_of_bins
        bins = np.arange(start=histogr_range[0], stop=histogr_range[1],
                         step=step)
        histo = pd.DataFrame(columns=bins)
        histo.loc[0] = np.zeros(len(histo.columns))
        for displ, angle in zip(displacements.values, angles):
            if not np.isnan(angle):
                _bin = cls._get_correct_bin_(bins, step, angle)
                histo[_bin] = histo[_bin] + np.linalg.norm(displ)

        for curr_bin in bins:
            result[f"{histogr_prefix}_{str(int(curr_bin))}"] = histo[curr_bin]

        return result

    @classmethod
    def _get_correct_bin_(cls, bins, step, angle):
        rounded_angle = round(angle, 4)
        if rounded_angle < 0:
            curr_angle = 360 + rounded_angle
        elif rounded_angle == 360:
            curr_angle = 0
        else:
            curr_angle = rounded_angle

        # check that angle can be added to histogram
        assert curr_angle <= bins[-1]+step, f"angle is {str(curr_angle)}"

        for _bin in bins:
            if _bin <= curr_angle < _bin + step:
                return _bin
