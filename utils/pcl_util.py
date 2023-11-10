import numpy as np

from misc import config


def pcl_to_bev(pcl):
    cfg = config.load()
    pcl_npa = pcl.numpy()
    mask = np.where((pcl_npa[:, 0] >= cfg.range_x[0]) & (pcl_npa[:, 0] <= cfg.range_x[1]) &
                    (pcl_npa[:, 1] >= cfg.range_y[0]) & (pcl_npa[:, 1] <= cfg.range_y[1]) &
                    (pcl_npa[:, 2] >= cfg.range_z[0]) & (pcl_npa[:, 2] <= cfg.range_z[1]))
    pcl_npa = pcl_npa[mask]

    # compute bev-map discretization by dividing x-range by the bev-image height
    bev_discrete = (cfg.range_x[1] - cfg.range_x[0]) / cfg.bev_height

    # create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates
    pcl_cpy = np.copy(pcl_npa)
    pcl_cpy[:, 0] = np.int_(np.floor(pcl_cpy[:, 0] / bev_discrete))

    # transform all metrix y-coordinates as well but center the forward-facing x-axis in the middle of the image
    pcl_cpy[:, 1] = np.int_(np.floor(pcl_cpy[:, 1] / bev_discrete) + (cfg.bev_width + 1) / 2)

    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    pcl_cpy[:, 2] = pcl_cpy[:, 2] - cfg.range_z[0]

    # re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then by decreasing height
    idx_height = np.lexsort((-pcl_cpy[:, 2], pcl_cpy[:, 1], pcl_cpy[:, 0]))
    lidar_pcl_hei = pcl_cpy[idx_height]

    # extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
    _, idx_height_unique = np.unique(lidar_pcl_hei[:, 0:2], axis=0, return_index=True)
    lidar_pcl_hei = lidar_pcl_hei[idx_height_unique]

    # assign the height value of each unique entry in lidar_top_pcl to the height map and
    # make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
    height_map = np.zeros((cfg.bev_height + 1, cfg.bev_width + 1))
    height_map[np.int_(lidar_pcl_hei[:, 0]), np.int_(lidar_pcl_hei[:, 1])] = lidar_pcl_hei[:, 2] / float(
        np.abs(cfg.range_z[1] - cfg.range_z[0]))

    # sort points such that in case of identical BEV grid coordinates, the points in each grid cell are arranged based on their intensity
    pcl_cpy[pcl_cpy[:, 2] > 1.0, 2] = 1.0
    idx_intensity = np.lexsort((-pcl_cpy[:, 2], pcl_cpy[:, 1], pcl_cpy[:, 0]))
    pcl_cpy = pcl_cpy[idx_intensity]

    # only keep one point per grid cell
    _, indices = np.unique(pcl_cpy[:, 0:2], axis=0, return_index=True)
    lidar_pcl_int = pcl_cpy[indices]

    # create the intensity map
    intensity_map = np.zeros((cfg.bev_height + 1, cfg.bev_width + 1))
    intensity_map[np.int_(lidar_pcl_int[:, 0]), np.int_(lidar_pcl_int[:, 1])] = lidar_pcl_int[:, 2] / (
            np.amax(lidar_pcl_int[:, 2]) - np.amin(lidar_pcl_int[:, 2]))

    # Compute density layer of the BEV map
    density_map = np.zeros((cfg.bev_height + 1, cfg.bev_width + 1))
    _, _, counts = np.unique(lidar_pcl_int[:, 0:2], axis=0, return_index=True, return_counts=True)
    normalized_counts = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    density_map[np.int_(lidar_pcl_int[:, 0]), np.int_(lidar_pcl_int[:, 1])] = normalized_counts

    bev_map = np.zeros((3, cfg.bev_height, cfg.bev_width))
    bev_map[2, :, :] = density_map[:cfg.bev_height, :cfg.bev_width]  # r_map
    bev_map[1, :, :] = height_map[:cfg.bev_height, :cfg.bev_width]  # g_map
    bev_map[0, :, :] = intensity_map[:cfg.bev_height, :cfg.bev_width]  # b_map

    bev_map = (np.transpose(bev_map, (1, 2, 0)) * 255).astype(np.uint8)

    return bev_map
