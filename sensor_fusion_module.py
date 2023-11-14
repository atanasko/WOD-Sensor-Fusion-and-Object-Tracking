import os
import sys
import glob
import argparse
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
import utils.wod_reader as wod_reader
import utils.pcl_util as pcl_util
import utils.visualizer as visualizer
from waymo_open_dataset import v2
from waymo_open_dataset.v2.perception.utils import lidar_utils as _lidar_utils

from ultralytics import YOLO

from perception_module import Lidar
from perception_module import Camera
from perception_module import EKF
from track_management_module import Tracker
from data_association_module import Association
from perception_module import LidarMeasurement
from perception_module import CameraMeasurement

import misc.config as config

cfg = config.load()


def update(meas_list):
    while association.A.shape[0] > 0 and association.A.shape[1] > 0:
        idx_track, idx_meas = association.get_nearest_track_measurement_indices()
        if np.isnan(idx_track):
            print('---no more associations---')
            break

        association.unassigned_tracks.remove(association.unassigned_tracks[idx_track])
        association.unassigned_meas.remove(association.unassigned_meas[idx_meas])

        track = tracker.track_list[idx_track]
        meas = meas_list[idx_meas]

        # check visibility, only update tracks in fov
        if not meas.sensor.in_fov(track.x):
            continue

        EKF.update(track, meas)
        track.increase_score()
        track.update_state()


if __name__ == '__main__':
    print('Sensor fusion start!')

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--dataset_root_dir', help="PATH is dataset root dir path")
    # parser.add_argument('-v', dest='verbose', action='store_true')
    args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])

    # Load a pretrained YOLOv8n model
    bev_model = YOLO('model/bev.pt')
    yolo_model = YOLO('model/yolov8n.pt')

    dataset_dir = args.dataset_root_dir + "/data/testing"
    context_names = [os.path.splitext(os.path.basename(name))[0] for name in glob.glob(dataset_dir + "/lidar/*.*")]

    context_name = context_names[12]

    # Loop over all the scenes dataframes?
    # Read camera calibration
    laser_name = 1
    camera_name = 1
    camera_calibration_df = wod_reader.read_camera_calibration_df(dataset_dir, context_name, camera_name)
    camera_calibration_dict = camera_calibration_df.head(npartitions=-1)
    camera_calibration = v2.CameraCalibrationComponent.from_dict(camera_calibration_dict)

    f_i = camera_calibration.intrinsic.f_u
    f_j = camera_calibration.intrinsic.f_v
    c_i = camera_calibration.intrinsic.c_u
    c_j = camera_calibration.intrinsic.c_v

    Camera.f_i = float(f_i)
    Camera.f_j = float(f_j)
    Camera.c_i = float(c_i)
    Camera.c_j = float(c_j)

    Camera.sens_to_vehicle = np.asmatrix(camera_calibration.extrinsic.transform[0].reshape(4, 4))
    # Camera.sens_to_vehicle = np.asmatrix(camera_calibration.extrinsic.transform[0].reshape(4, 4))
    Camera.sens_to_vehicle = np.asmatrix(tf.reshape(tf.constant(list(camera_calibration.extrinsic.transform[0]), dtype=tf.float32), [4, 4]).numpy())
    Camera.veh_to_sens = np.linalg.inv(Camera.sens_to_vehicle)

    EKF = EKF()
    tracker = Tracker()
    association = Association()

    lidar_meas_list = []
    cam_meas_list = []

    lidar_df = wod_reader.read_lidar_df(dataset_dir, context_name, laser_name)
    lidar_calibration_df = wod_reader.read_lidar_calibration_df(dataset_dir, context_name, laser_name)
    lidar_pose_df = wod_reader.read_lidar_pose_df(dataset_dir, context_name, laser_name)
    cam_img_df = wod_reader.read_cam_img_df(dataset_dir, context_name, laser_name)
    camera_calibration_df = wod_reader.read_camera_calibration_df(dataset_dir, context_name, laser_name)
    vehicle_pose_df = wod_reader.read_vehicle_pose_df(dataset_dir, context_name)

    df = lidar_df[lidar_df['key.laser_name'] == laser_name]
    df = v2.merge(df, lidar_calibration_df)
    df = v2.merge(df, lidar_pose_df)
    df = v2.merge(df, cam_img_df[cam_img_df['key.camera_name'] == camera_name])
    df = v2.merge(df, camera_calibration_df)
    df = v2.merge(df, vehicle_pose_df)

    for i, (_, r) in enumerate(df.iterrows()):
        # ########## LIDAR detect ##########
        lidar = v2.LiDARComponent.from_dict(r)
        lidar_calibration = v2.LiDARCalibrationComponent.from_dict(r)
        lidar_pose = v2.LiDARPoseComponent.from_dict(r)
        vehicle_pose = v2.VehiclePoseComponent.from_dict(r)

        pcl = _lidar_utils.convert_range_image_to_point_cloud(lidar.range_image_return1, lidar_calibration,
                                                              lidar_pose.range_image_return1, vehicle_pose)
        bev_img = pcl_util.pcl_to_bev(pcl)
        bev_img = cv2.rotate(bev_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img = Image.fromarray(bev_img).convert('RGB')


        lidar_detections = bev_model.predict(img)  # results list
        visualizer.vis_bev_image(bev_img, lidar_detections[0].boxes)
        for box, cls in zip(lidar_detections[0].boxes.xywh, lidar_detections[0].boxes.cls):
            if float(cls) == 1:
                z = [float(box[0]), float(box[1]), 1, float(box[2]), float(box[3]), 1.5, 0]
                ratio_x = (cfg.range_x[1] - cfg.range_x[0]) / cfg.bev_width
                z[0] *= ratio_x
                z[1] = (- z[1] + cfg.bev_height / 2) * ratio_x
                z[3] *= ratio_x
                z[4] *= ratio_x
                meas = LidarMeasurement(z)
                lidar_meas_list.append(meas)

        # ########## CAMERA detect ##########
        cam = v2.CameraImageComponent.from_dict(r)
        cam_calibration = v2.CameraCalibrationComponent.from_dict(r)
        img = Image.open(BytesIO(cam.image))
        camera_detections = yolo_model.predict(img)  # results list
        for box, cls in zip(camera_detections[0].boxes.xywh, lidar_detections[0].boxes.cls):
            if float(cls) == 1:
                z = [float(box[0]), float(box[1]), 0, 0, 0, 0, 0]
                meas = CameraMeasurement(z)
                cam_meas_list.append(meas)

        # ########## PREDICT ##########
        for track in tracker.track_list:
            EKF.predict(track)

        if lidar_meas_list:
            # ########## UPDATE LIDAR ##########
            association.associate(tracker.track_list, lidar_meas_list)
            update(lidar_meas_list)

            # ########## UPDATE UNASSIGNED TRACKS ##########
            sensor = lidar_meas_list[0].sensor
            tracker.update_unassigned_tracks(association.unassigned_tracks, sensor)

            # ########## PROCESS UNASSIGNED MEASUREMENTS ##########
            # only initialize with lidar measurements
            tracker.process_unassigned_meas(lidar_meas_list, association.unassigned_meas)
            association.unassigned_meas.clear()

        if cam_meas_list:
            # ########## UPDATE CAMERA ##########
            association.associate(tracker.track_list, cam_meas_list)
            update(cam_meas_list)

            # ########## UPDATE UNASSIGNED TRACKS ##########
            sensor = cam_meas_list[0].sensor
            tracker.update_unassigned_tracks(association.unassigned_tracks, sensor)

        # ########## DELETE TRACKS ##########
        tracker.process_delete_track_candidates()

        # im_array = camera_detections[0].plot()  # plot a BGR numpy array of predictions
        # im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        # im.show()  # show image

        visualizer.vis_bev_image_1(bev_img, lidar_detections[0].boxes)

        visualizer.vis_cam_img(img, tracker.track_list, Camera())
