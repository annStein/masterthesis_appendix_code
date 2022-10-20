import statistics

import numpy as np
import pandas as pd

from definitions import FRAMERATE_JUMP_THESHOLD
from helper.interpolation import interpolate_nan_values
from models.input_data import InputData
from preprocessors.preprocessor_base import Preprocessor


class PreprocessorFramedrops(Preprocessor):
    """
    Interpolates framedrops. Framedrops are identified by jumps in timestamps.
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
                cls.logger.info(f'Interpolate framedrops of {landmark_filename} ...')
                result = landmark_data.landmarks.copy()

                # get framerates and median framerate
                timestamps = landmark_data.landmarks['timestamp']
                framerate = 1000000 / np.diff(timestamps)
                if len(framerate) > 0:
                    median_framrate = statistics.median(framerate)

                    # set threshold for correct framerate
                    frame_thresh = FRAMERATE_JUMP_THESHOLD

                    # get indices of framedrops
                    framedrop_idx = [idx for idx, element in enumerate(framerate) if
                                     element <= median_framrate - median_framrate * frame_thresh]
                    for idx in framedrop_idx:
                        # add nan values at framesdrops
                        row_to_insert = {}
                        for col in result.columns:
                            row_to_insert[col] = np.nan
                        row = pd.DataFrame(row_to_insert, index=[idx + 1])
                        result = pd.concat(
                            [result.iloc[:idx + 1], row, result.iloc[idx + 1:]]).reset_index(drop=True)

                    # interpolate nan values
                    for col in result.columns:
                        result[col] = interpolate_nan_values(result[col])

                    landmark_data.landmarks = result
                    cls.logger.info(f'Interpolate framedrops of {landmark_filename} complete.')
            except Exception:
                cls.logger.exception(f'Exception during processing of {landmark_filename}')
                continue

        return input_data
