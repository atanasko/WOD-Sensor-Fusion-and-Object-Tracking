import numpy as np
from enum import Enum
import misc.params as params

score_delta = 1. / params.n_last_frames
init_threshold = 1. / params.n_last_frames


class State(Enum):
    INITIALIZED = 1
    TENTATIVE = 2
    CONFIRMED = 3


class Track:
    def __init__(self, id, meas):
        self.id = id
        self.score = score_delta
        self.state = State.INITIALIZED

        self.width = meas.width
        self.length = meas.length
        self.height = meas.height
        self.heading = 0

        pos_sens = np.ones((4, 1))
        pos_sens[0:3] = meas.z[0:3]
        pos_veh = meas.sensor.sens_to_veh * pos_sens

        # Track in vehicle coordinates
        self.x = np.zeros((6, 1))
        self.x[0:3] = pos_veh[0:3]

        M_rot = meas.sensor.sens_to_veh[0:3, 0:3]
        P_pos = M_rot * meas.R * np.transpose(M_rot)

        # |P_pos |   0   |
        # ----------------
        # |   0  | P_veh |
        # Lidar can't measure velocity so estimation error covariance for vehicle entries can be initialized with high
        # values to reflect high uncertainty in velocity
        sigma_vx = 50
        sigma_vy = 50
        sigma_vz = 5
        P_veh = np.matrix([[sigma_vx ** 2, 0, 0],
                           [0, sigma_vy ** 2, 0],
                           [0, 0, sigma_vz ** 2]])

        self.P = np.zeros((6, 6))
        self.P[0:3, 0:3] = P_pos
        self.P[3:6, 3:6] = P_veh

    def set_x(self, x):
        self.x = x

    def set_P(self, P):
        self.P = P

    def increase_score(self):
        if self.score < 1:
            self.score += score_delta

    def decrease_score(self):
        self.score -= score_delta

    def update_state(self):
        if init_threshold < self.score < params.confirm_threshold:
            self.state = State.TENTATIVE
        if self.score >= params.confirm_threshold:
            self.state = State.CONFIRMED


class Tracker:
    def __init__(self):
        self.track_list = []

    def init_track(self, meas):
        last_id = self.track_list[len(self.track_list) - 1].id if len(self.track_list) > 0 else 0
        track = Track(last_id + 1, meas)
        self.track_list.append(track)

    def process_delete_track_candidates(self):
        for track in self.track_list:
            if (((track.state == State.INITIALIZED or track.state == State.TENTATIVE)
                 and (track.score < params.delete_tentative_threshold
                      or (track.P[0, 0] > params.max_P or track.P[1, 1] > params.max_P)))
                    or ((track.state == State.CONFIRMED)
                        and (track.score < params.delete_confirmed_threshold
                             or (track.P[0, 0] > params.max_P or track[1, 1] > params.max_P)))):
                self.track_list.remove(track)

    def update_unassigned_tracks(self, unassigned_tracks, sensor):
        for i in unassigned_tracks:
            track = self.track_list[i]
            # check visibility
            if sensor.in_fov(track.x):
                track.decrease_score()

    def process_unassigned_meas(self, lidar_meas_list, unassigned_meas):
        for idx in unassigned_meas:
            self.init_track(lidar_meas_list[idx])
