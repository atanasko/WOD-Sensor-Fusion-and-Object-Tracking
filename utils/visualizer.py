import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

colormap = {1: "red", 2: "blue", 3: "yellow", 4: "yellow"}


def vis_cam_img(img, track_list, camera):
    ax = plt.subplot()
    # Iterate over the individual labels.
    for track in track_list:
        xv = np.ones((4, 1))
        xv[0:3] = track.x[0:3]
        xs = camera.veh_to_sens * xv
        # Draw the object bounding box.
        ratio_i = 1920 / 50
        ratio_j = 1280 / 50
        i = - xs[1] * ratio_i
        j = - xs[0] * ratio_j + 1280
        # size_x = track.x[3]
        # size_y = track.x[4]
        size_x = 100
        size_y = 100
        ax.add_patch(patches.Rectangle(
            xy=(i - 0.5 * size_x, j - 0.5 * size_y),
            width=size_x,
            height=size_y,
            linewidth=1,
            edgecolor=colormap[1],
            facecolor='none'))

    plt.imshow(img)
    plt.show()
