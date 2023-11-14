import numpy as np

import misc.params as params


class Sensor:
    dimm = None

    fov = None
    sens_to_veh = None
    veh_to_sens = None

    def __init__(self):
        pass

    def get_hx(self, x):
        ...

    def get_H(self, x):
        ...

    def in_fov(self, x):
        # check if an object x can be seen by this sensor
        pos_veh = np.ones((4, 1))  # homogeneous coordinates
        pos_veh[0:3] = x[0:3]
        pos_sens = self.veh_to_sens * pos_veh  # transform from vehicle to sensor coordinates

        theta = np.arctan(pos_sens[1] / pos_sens[0])

        return self.fov[0] < float(theta) < self.fov[1]


class Lidar(Sensor):
    dimm = 3

    def __init__(self):
        # transformation sensor to vehicle coordinates equals identity matrix because
        # lidar detections are already in vehicle coordinates
        super().__init__()

        self.sens_to_veh = np.matrix(np.identity(4))
        self.veh_to_sens = np.linalg.inv(self.sens_to_veh)

        self.fov = [-np.pi / 2, np.pi / 2]

    def get_hx(self, x):
        pos_veh = np.ones((4, 1))
        pos_veh[0:3] = x[0:3]
        pos_sens = self.veh_to_sens * pos_veh

        return pos_sens[0:3]

    def get_H(self, x):
        if x[0] == 0:
            raise NameError('Jacobian not defined for x[0]=0!')

        H = np.matrix(np.zeros((self.dimm, 6)))
        R = self.sens_to_veh[0:3, 0:3]
        H[0:3, 0:3] = R

        return H


class Camera(Sensor):
    dimm = 2

    f_i = None  # focal length i-coordinate
    f_j = None  # focal length j-coordinate
    c_i = None  # principal point i-coordinate
    c_j = None  # principal point j-coordinate

    def __init__(self):
        super().__init__()

        # self.sens_to_vehicle = np.matrix(np.identity(4))
        # self.veh_to_sens = np.linalg.inv(self.sens_to_vehicle)
        self.fov = [-0.35, 0.35]

    def get_hx(self, x):
        # check and print error message if dividing by zero
        if x[0] == 0:
            raise NameError('Jacobian not defined for x[0]=0!')

        hx = np.zeros((2, 1))

        pos_veh = np.ones((4, 1))  # homogeneous coordinates
        pos_veh[0:3] = x[0:3]
        pos_sens = self.veh_to_sens * pos_veh  # transform from vehicle to camera coordinates

        hx[0, 0] = self.c_i - self.f_i * (pos_sens[1] / pos_sens[0])
        hx[1, 0] = self.c_i - self.f_i * (pos_sens[2] / pos_sens[0])

        return hx

    def get_H(self, x):
        R = self.veh_to_sens[0:3, 0:3]  # rotation
        T = self.veh_to_sens[0:3, 3]  # translation

        if R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0] == 0:
            raise NameError('Jacobian not defined for this x!')

        # [u * v]' = u'v + uv'
        Hj = np.matrix(np.zeros((self.dimm, 6)))
        Hj[0, 0] = self.f_i * (-R[1, 0] / (R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0])
                               + R[0, 0] * (R[1, 0] * x[0] + R[1, 1] * x[1] + R[1, 2] * x[2] + T[1])
                               / ((R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0]) ** 2))
        Hj[1, 0] = self.f_j * (-R[2, 0] / (R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0])
                               + R[0, 0] * (R[2, 0] * x[0] + R[2, 1] * x[1] + R[2, 2] * x[2] + T[2])
                               / ((R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0]) ** 2))
        Hj[0, 1] = self.f_i * (-R[1, 1] / (R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0])
                               + R[0, 1] * (R[1, 0] * x[0] + R[1, 1] * x[1] + R[1, 2] * x[2] + T[1])
                               / ((R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0]) ** 2))
        Hj[1, 1] = self.f_j * (-R[2, 1] / (R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0])
                               + R[0, 1] * (R[2, 0] * x[0] + R[2, 1] * x[1] + R[2, 2] * x[2] + T[2])
                               / ((R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0]) ** 2))
        Hj[0, 2] = self.f_i * (-R[1, 2] / (R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0])
                               + R[0, 2] * (R[1, 0] * x[0] + R[1, 1] * x[1] + R[1, 2] * x[2] + T[1])
                               / ((R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0]) ** 2))
        Hj[1, 2] = self.f_j * (-R[2, 2] / (R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0])
                               + R[0, 2] * (R[2, 0] * x[0] + R[2, 1] * x[1] + R[2, 2] * x[2] + T[2])
                               / ((R[0, 0] * x[0] + R[0, 1] * x[1] + R[0, 2] * x[2] + T[0]) ** 2))

        return Hj


class Measurement:
    sensor = None

    def __init__(self):
        pass


class LidarMeasurement(Measurement):
    sensor = Lidar()

    def __init__(self, z):
        super().__init__()

        self.z = np.zeros((self.sensor.dimm, 1))
        self.z[0] = z[0]
        self.z[1] = z[1]
        self.z[2] = z[2]

        sigma_lidar_x = params.sigma_lidar_x
        sigma_lidar_y = params.sigma_lidar_y
        sigma_lidar_z = params.sigma_lidar_z
        self.R = np.matrix([[sigma_lidar_x ** 2, 0, 0],
                            [0, sigma_lidar_y ** 2, 0],
                            [0, 0, sigma_lidar_z ** 2]])

        self.width = z[3]
        self.length = z[4]
        self.height = z[5]
        self.yaw = z[6]


class CameraMeasurement(Measurement):
    sensor = Camera()

    def __init__(self, z):
        super().__init__()

        self.z = np.zeros((self.sensor.dimm, 1))
        self.z[0] = z[0]
        self.z[1] = z[1]

        sigma_cam_i = params.sigma_cam_i  # load params
        sigma_cam_j = params.sigma_cam_j
        self.R = np.matrix([[sigma_cam_i ** 2, 0],
                            [0, sigma_cam_j ** 2]])


class EKF:
    def __init__(self):
        dim_state = 4

    def F(self):
        dt = 1
        return np.matrix([[1, 0, 0, dt, 0, 0],
                          [0, 1, 0, 0, dt, 0],
                          [0, 0, 1, 0, 0, dt],
                          [0, 0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]])

    def Q(self):
        dt = 1
        q = params.q
        q1 = q * dt
        q2 = 1 / 2 * q * (dt ** 2)
        q3 = 1 / 3 * q * (dt ** 3)

        return np.matrix([[q3, 0, 0, q2, 0, 0],
                          [0, q3, 0, 0, q2, 0],
                          [0, 0, q3, 0, 0, q2],
                          [q2, 0, 0, q1, 0, 0],
                          [0, q2, 0, 0, q1, 0],
                          [0, 0, q2, 0, 0, q1]])

    def predict(self, track):
        F = self.F()
        x = F * track.x
        P = F * track.P * F.transpose() + self.Q()

        track.set_x(x)
        track.set_P(P)

    def update(self, track, meas):
        Hj = meas.sensor.get_H(track.x)
        gamma = meas.z - meas.sensor.get_hx(track.x)
        S = Hj * track.P * Hj.transpose() + meas.R
        K = track.P * Hj.transpose() * np.linalg.inv(S)

        x = track.x + K * gamma
        # I = np.identity(meas.sensor.dimm)
        I = np.identity(6)
        P = (I - K * Hj) * track.P

        track.set_x(x)
        track.set_P(P)
