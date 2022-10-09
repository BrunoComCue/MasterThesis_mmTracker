import numpy as np
from .. import motion_model
from .. import life as life_manager
from ..update_info_data import UpdateInfoData
from ..frame_data import FrameData
from ..data_protos import BBox
from math import sqrt


class Tracklet:
    def __init__(self, configs, id, bbox: BBox, det_type, frame_index, time_stamp=None, aux_info=None):
        self.id = id
        self.time_stamp = time_stamp
        self.asso = configs['running']['asso']

        self.configs = configs
        self.det_type = det_type
        self.aux_info = aux_info

        # create KF and CV motion models:
        self.motion_model_kf = motion_model.KalmanFilterMotionModel(
                bbox=bbox, inst_type=self.det_type, time_stamp=time_stamp, covariance=configs['running']['covariance'])
        self.motion_model_cv = motion_model.NaiveMotionModel(bbox=bbox, velo=aux_info['velo'], inst_type=self.det_type,
                time_stamp=time_stamp)

        # life and death management:
        self.life_manager = life_manager.HitManager(configs, frame_index)

        # store the score for the latest bbox:
        self.latest_score = bbox.s

    def weight(self, kf_model, cv_model):
        result = kf_model
        d = sqrt(((result.x + cv_model.x)/2)**2 + ((result.y + cv_model.y)/2)**2 + ((result.z + cv_model.z)/2)**2)
        if 0 <= d < 3000:
            result.x = 0.7 * cv_model.x + 0.3 * result.x
            result.y = 0.7 * cv_model.y + 0.3 * result.y
            result.z = 0.7 * cv_model.z + 0.3 * result.z
        elif 3000 <= d < 5000:
            result.x = 0.3 * cv_model.x + 0.7 * result.x
            result.y = 0.3 * cv_model.y + 0.7 * result.y
            result.z = 0.3 * cv_model.z + 0.7 * result.z
        else:
            result.x = 0.1 * cv_model.x + 0.9 * result.x
            result.y = 0.1 * cv_model.y + 0.9 * result.y
            result.z = 0.1 * cv_model.z + 0.9 * result.z
        return result

    def predict(self, time_stamp=None, is_key_frame=True):
        """ in the prediction step, the motion model predicts the state of bbox
            the other components have to be synced
            the result is a BBox
            the usage of time_stamp is optional, only if you use velocities
        """
        # predicted bbox resulting from KF model:
        result_kf = self.motion_model_kf.get_prediction(time_stamp=time_stamp)
        # predicted bbox resulting from CV model:
        result_cv = self.motion_model_cv.get_prediction(time_stamp=time_stamp)
        self.life_manager.predict(is_key_frame=is_key_frame)
        self.latest_score = self.latest_score * 0.01
        # predicted bbox score (same for both):
        result_kf.s = self.latest_score
        result_cv.s = self.latest_score
        return self.weight(result_kf, result_cv)

    def update(self, update_info: UpdateInfoData):
        """ update the state of the tracklet
        """
        self.latest_score = update_info.bbox.s
        is_key_frame = update_info.aux_info['is_key_frame']

        # only the direct association update the motion model
        # this updates both KF and CV motion models:
        if update_info.mode == 1 or update_info.mode == 3:
            self.motion_model_kf.update(update_info.bbox, update_info.aux_info)
            self.motion_model_cv.update(update_info.bbox, update_info.aux_info)
        else:
            pass
        self.life_manager.update(update_info, is_key_frame)
        return

    def get_state(self):
        """ current state of the tracklet
        """
        # get current state of the bbox using both motion models:
#        result_kf = self.motion_model_kf.get_state()
#        result_cv = self.motion_model_cv.get_state()

#        result_kf.s = self.latest_score
#        result_cv.s = self.latest_score
#        return self.weight(result_kf, result_cv)
        result = self.motion_model_kf.get_state()  # both models return the same: the current bbox
        result.s = self.latest_score
        return result
    def valid_output(self, frame_index):
        return self.life_manager.valid_output(frame_index)

    def death(self, frame_index):
        return self.life_manager.death(frame_index)

    def state_string(self, frame_index):
        """ the string describes how we get the bbox (e.g. by detection or motion model prediction)
        """
        return self.life_manager.state_string(frame_index)

    def compute_innovation_matrix(self):
        """ compute the innovation matrix for association with mahalonobis distance
        """
        return self.motion_model_kf.compute_innovation_matrix()

    def sync_time_stamp(self, time_stamp):
        """ sync the time stamp for motion model
        """
        self.motion_model_kf.sync_time_stamp(time_stamp)
        self.motion_model_cv.sync_time_stamp(time_stamp)
        return