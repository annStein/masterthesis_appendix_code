import numpy as np

from masterthesis_appendix_code.constants.definitions import BODY_PARTS
from helper.interpolation import interpolate_nan_values
from models.input_data import InputData
from preprocessors.preprocessor_base import Preprocessor


class PreprocessorInvalidLandmarks(Preprocessor):
    """
    Preprocessor to set landmarks where x, y and z are 0 to nan and optionally interpolates those
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
        interpolate: bool = True
    ) -> InputData:

        for landmark_filename, landmark_data in input_data.kinect_landmark_data_by_filename.items():
            try:
                cls.logger.info(f'Setting invalid landmarks of {landmark_filename} to nan...')
                result = landmark_data.landmarks.copy()
                for part in BODY_PARTS:
                    for index in range(len(landmark_data.landmarks)):
                        if round(float(landmark_data.landmarks[part + "_x"][index]), 4) == 0.0000 and \
                                round(float(landmark_data.landmarks[part + "_y"][index]), 4) == 0.0000 and \
                                round(float(landmark_data.landmarks[part + "_z"][index]), 4) == 0.0000:
                            result[part + "_x"][index] = np.nan
                            result[part + "_y"][index] = np.nan
                            result[part + "_z"][index] = np.nan
                    if interpolate:
                        result[part + "_x"] = interpolate_nan_values(np.array(result[part + "_x"]))
                        result[part + "_y"] = interpolate_nan_values(np.array(result[part + "_y"]))
                        result[part + "_z"] = interpolate_nan_values(np.array(result[part + "_z"]))
                landmark_data.landmarks = result
                cls.logger.info(f'Setting invalid landmarks of {landmark_filename} to nan complete.')
            except Exception:
                cls.logger.exception(f'Exception during processing of {landmark_filename}')
                continue

        return input_data
