import pandas as pd

from helper.landmark import get_landmark


def move_points_to_origin(landmark_data, body_parts, reference_point):
    assert len(reference_point) == len(landmark_data)
    result = pd.DataFrame()
    for part in body_parts:
        result[part + "_x"] = landmark_data[part + "_x"] - reference_point.x
        result[part + "_y"] = landmark_data[part + "_y"] - reference_point.y
        result[part + "_z"] = landmark_data[part + "_z"] - reference_point.z
        result[part + "_confidence"] = landmark_data[part + "_confidence"]
    return result

def norm_single_lm_by_size(landmark_data, landmark_name, reference_point, reference_distance):
    assert len(reference_point) == len(landmark_data)
    _lm_bef = get_landmark(landmark_data, landmark_name)
    _lm_normed = pd.DataFrame()
    _lm_normed["x"] = (_lm_bef.x - reference_point.x) / reference_distance
    _lm_normed["y"] = (_lm_bef.y - reference_point.y) / reference_distance
    _lm_normed["z"] = (_lm_bef.z - reference_point.z) / reference_distance
    return _lm_normed