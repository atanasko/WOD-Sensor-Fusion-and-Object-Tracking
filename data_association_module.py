import numpy as np
from scipy.stats.distributions import chi2

import misc.params as params


class Association:
    def __init__(self):
        self.A = np.matrix([])  # Association matrix
        self.unassigned_tracks = []
        self.unassigned_meas = []

    def get_nearest_track_measurement_indices(self):
        if np.min(self.A) == np.inf:
            return np.nan, np.nan
        i, j = np.unravel_index(np.argmin(self.A, axis=None), self.A.shape)
        self.A = np.delete(self.A, i, axis=0)
        self.A = np.delete(self.A, j, axis=1)

        return i, j

    def associate(self, track_list, meas_list):
        N = len(track_list)
        M = len(meas_list)

        self.unassigned_tracks = list(range(N))
        self.unassigned_meas = list(range(M))

        self.A = np.inf * np.ones([N, M])

        for i in range(N):
            track = track_list[i]
            for j in range(M):
                meas = meas_list[j]
                MHD = self.MHD(track, meas)
                if self.in_gate(meas.sensor, MHD):
                    self.A[i, j] = MHD

    def MHD(self, track, meas):
        Hj = meas.sensor.get_H(track.x)

        gamma = meas.z - Hj * track.x
        S = Hj * track.P * Hj.transpose() + meas.R

        return gamma.transpose() * np.linalg.inv(S) * gamma

    def in_gate(self, sensor, MHD):
        return MHD < chi2.ppf(params.gating_threshold, df=sensor.dimm)
