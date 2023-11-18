import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

import misc.config as config
import cv2
from PIL import Image

colormap = {1: "red", 2: "blue", 3: "yellow", 4: "yellow"}


def vis_cam_img(img, track_list, camera):
    ax = plt.subplot()
    # Iterate over the individual labels.
    for track in track_list:
        xv = np.ones((4, 1))
        xv[0:3] = track.x[0:3]

        xi = camera.get_hx(xv)

        # Draw the object bounding box.
        i = xi[0]
        j = xi[1]

        size_x = track.length * 1920 / 50 * 2
        size_y = track.width * 1280 / 50 * 2
        ax.add_patch(patches.Rectangle(
            xy=(i, j),
            width=size_x,
            height=size_y,
            linewidth=1,
            edgecolor=colormap[1],
            facecolor='none'))

    plt.imshow(img)
    plt.show()


def vis_bev_image(bev_img, boxes):
    cfg = config.load()

    img = Image.fromarray(bev_img).convert('RGB')

    ax = plt.subplot()
    # Iterate over the individual labels.
    for box in boxes:
        # Draw the object bounding box.
        box_xywh = box.xywh[0]
        x = float(box_xywh[0])
        y = float(box_xywh[1])

        size_x = float(box_xywh[2])
        size_y = float(box_xywh[3])

        ax.add_patch(patches.Rectangle(
            xy=(x - 0.5 * size_x,
                y - 0.5 * size_y),
            width=size_x,
            height=size_y,
            linewidth=1,
            edgecolor=colormap[1],
            facecolor='none'))

    plt.imshow(img)
    plt.show()


def vis_bev_image_1(bev_img, boxes):
    ax = plt.subplot()

    img = Image.fromarray(bev_img).convert('RGB')
    img = cv2.rotate(np.array(img), cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Iterate over the individual labels.
    for box in boxes:
        # Draw the object bounding box.
        box_xywh = box.xywh[0]
        x = float(box_xywh[0])
        y = float(box_xywh[1])

        original_height, original_width = 640, 640
        rotated_width, rotated_height = 1280, 1920
        cx, cy = original_width // 2, original_height // 2
        theta = np.radians(-90)
        scale_width = rotated_width / original_width
        scale_height = rotated_height / original_height

        x1 = int(np.cos(theta) * (x - cx) - np.sin(theta) * (y - cy) + cx)
        y1 = int(np.sin(theta) * (x - cx) + np.cos(theta) * (y - cy) + cy)

        size_x = float(box_xywh[2])
        size_y = float(box_xywh[3])

        ax.add_patch(patches.Rectangle(
            xy=(x1,
                y1),
            width=20,
            height=20,
            linewidth=1,
            edgecolor=colormap[1],
            facecolor='none'))

    plt.imshow(img)
    plt.show()


def vis_cam_image_bev_bboxes(img, boxes):
    ax = plt.subplot()

    # img1 = cv2.rotate(np.array(img), cv2.ROTATE_90_CLOCKWISE)

    # Iterate over the individual labels.
    for box in boxes:
        # Draw the object bounding box.
        box_xywh = box.xywh[0]
        x = float(box_xywh[0])
        y = float(box_xywh[1])

        original_height, original_width = 640, 640
        # rotated_width, rotated_height = 1280, 1920
        rotated_width, rotated_height = 1920, 1280
        cx, cy = original_width // 2, original_height // 2
        theta = np.radians(-90)
        scale_width = rotated_width / original_width
        scale_height = rotated_height / original_height

        x1 = int(scale_width * (np.cos(theta) * (x - cx) - np.sin(theta) * (y - cy) + cx))
        y1 = int(scale_height * (np.sin(theta) * (x - cx) + np.cos(theta) * (y - cy) + cy))

        # x = x * 1920 / 640
        # y = y * 1280 / 640

        size_x = float(box_xywh[2])
        size_y = float(box_xywh[3])

        ax.add_patch(patches.Rectangle(
            xy=(x1,
                y1),
            width=size_x,
            height=size_y,
            linewidth=1,
            edgecolor=colormap[1],
            facecolor='none'))

    plt.imshow(img)
    plt.show()
