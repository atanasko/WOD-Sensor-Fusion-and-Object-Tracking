# general parameters
dim_state = 6  # process model dimension

q = 3  # process noise variable for Kalman filter Q

n_last_frames = 6
confirm_threshold = 0.8  # track score threshold to switch from 'tentative' to 'confirmed'
delete_tentative_threshold = 0.15
delete_confirmed_threshold = 0.6  # track score threshold to delete confirmed tracks

max_P = 3 ** 2  # delete track if covariance of px or py bigger than this

sigma_lidar_x = 0.1  # measurement noise standard deviation for lidar x position
sigma_lidar_y = 0.1  # measurement noise standard deviation for lidar y position
sigma_lidar_z = 0.1  # measurement noise standard deviation for lidar z position

sigma_cam_i = 5  # measurement noise standard deviation for image i coordinate
sigma_cam_j = 5  # measurement noise standard deviation for image j coordinate

gating_threshold = 0.995  # percentage of correct measurements that shall lie inside gate
