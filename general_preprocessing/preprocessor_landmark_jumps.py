import numpy as np
import pandas as pd

from definitions import CALIBRATION_JUMP_THRESHOLD, BODY_PARTS
from helper.interpolation import interpolate_nan_values
from models.input_data import InputData
from preprocessors.preprocessor_base import Preprocessor


class PreprocessorCalibrationJumps(Preprocessor):
    """
    Preprocessor to remove calibration jumps from Kinect landmarks
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
        **kwargs
    ) -> InputData:

        for landmark_filename, landmark_data in input_data.kinect_landmark_data_by_filename.items():
            try:
                cls.logger.info(f'Removing calibration jumps of {landmark_filename} ...')

                # result = cls._remove_outliers_per_dimension_(landmark_data.landmarks)
                result = cls._remove_outliers_overall_(landmark_data.landmarks)

                landmark_data.landmarks = result
                cls.logger.info(f'Removing calibration jumps of {landmark_filename} complete.')
            except Exception:
                cls.logger.exception(f'Exception during processing of {landmark_filename}')
                continue

        return input_data

    @classmethod
    def _remove_outliers_per_dimension_(cls, landmark_data):
        result = landmark_data.copy()
        x_coords = landmark_data.iloc[:, 1::4]
        y_coords = landmark_data.iloc[:, 2::4]
        z_coords = landmark_data.iloc[:, 3::4]
        coords = zip(x_coords, y_coords, z_coords)
        for x, y, z in coords:
            x_landmarks = np.copy(landmark_data[x]).astype(float)
            y_landmarks = np.copy(landmark_data[y]).astype(float)
            z_landmarks = np.copy(landmark_data[z]).astype(float)

            # -------------------- velocity -------------------- #
            x_velo = np.diff(x_landmarks).astype(float)
            y_velo = np.diff(y_landmarks).astype(float)
            z_velo = np.diff(z_landmarks).astype(float)

            # -------------------- filter landmark values -------------------- #
            thresh = CALIBRATION_JUMP_THRESHOLD
            x_outlier_idx_velo = [idx for idx, element in enumerate(x_velo) if abs(element) > thresh]
            y_outlier_idx_velo = [idx for idx, element in enumerate(y_velo) if abs(element) > thresh]
            z_outlier_idx_velo = [idx for idx, element in enumerate(z_velo) if abs(element) > thresh]

            # max. 45(?) consecutive frames could be outliers, TODO: check number
            # (consecutive) frames are handled as outliers if the velocity jump goes in 2 directions (e.g. first up, then down)
            frame_thresh = 45
            prev_idx_x, prev_idx_y, prev_idx_z = np.nan, np.nan, np.nan
            for idx_x in x_outlier_idx_velo:
                if (idx_x - prev_idx_x <= frame_thresh) and ((x_velo[idx_x] > 0 and x_velo[prev_idx_x] < 0) or (
                        x_velo[idx_x] < 0 and x_velo[prev_idx_x] > 0)):
                    x_landmarks[prev_idx_x + 1:idx_x + 1] = np.nan
                prev_idx_x = idx_x
            for idx_y in y_outlier_idx_velo:
                if (idx_y - prev_idx_y <= frame_thresh) and ((y_velo[idx_y] > 0 and y_velo[prev_idx_y] < 0) or (
                        y_velo[idx_y] < 0 and y_velo[prev_idx_y] > 0)):
                    y_landmarks[prev_idx_y + 1:idx_y + 1] = np.nan
                prev_idx_y = idx_y
            for idx_z in z_outlier_idx_velo:
                if (idx_z - prev_idx_z <= frame_thresh) and ((z_velo[idx_z] > 0 and z_velo[prev_idx_z] < 0) or (
                        z_velo[idx_z] < 0 and z_velo[prev_idx_z] > 0)):
                    z_landmarks[prev_idx_z + 1:idx_z + 1] = np.nan
                prev_idx_z = idx_z
            result[x] = interpolate_nan_values(x_landmarks)
            result[y] = interpolate_nan_values(y_landmarks)
            result[z] = interpolate_nan_values(z_landmarks)
        return result

    @classmethod
    def _remove_outliers_overall_(cls, landmark_data):
        result = landmark_data.copy()
        for part in BODY_PARTS:
            velo = {}
            velo["x"] = np.diff(landmark_data[part + "_x"])
            velo["y"] = np.diff(landmark_data[part + "_y"])
            velo["z"] = np.diff(landmark_data[part + "_z"])
            velo_df = pd.DataFrame(velo).astype(float)
            velo_total = np.linalg.norm(velo_df, axis=1)

            thresh = CALIBRATION_JUMP_THRESHOLD
            outlier_idx_velo = [idx for idx, element in enumerate(velo_total) if abs(element) > thresh]

            # max. 45(?) consecutive frames could be outliers, TODO: check number
            # (consecutive) frames are handled as outliers if the velocity jump goes in 2 directions (e.g. first up, then down)
            frame_thresh = 45
            prev_idx = np.nan
            for idx in outlier_idx_velo:
                if (idx - prev_idx <= frame_thresh) and ((velo_total[idx] > 0 and velo_total[prev_idx] < 0) or (
                        velo_total[idx] < 0 and velo_total[prev_idx] > 0)):
                    result[part + "_x"][prev_idx + 1:idx + 1] = np.nan
                    result[part + "_y"][prev_idx + 1:idx + 1] = np.nan
                    result[part + "_z"][prev_idx + 1:idx + 1] = np.nan
                prev_idx = idx

            # interpolate nan values
            result[part + "_x"] = interpolate_nan_values(result[part + "_x"])
            result[part + "_y"] = interpolate_nan_values(result[part + "_y"])
            result[part + "_z"] = interpolate_nan_values(result[part + "_z"])
        return result
