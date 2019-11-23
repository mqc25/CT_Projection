import numpy as np
import math
from numba import jit, prange

def calculate_extrinsic_matrix(distance, side='r'):
    d = distance/2.0
    sign = 1
    if side == 'l':
        sign = -1

    p = np.array(([d, 100, 0], [d, 0, -100], [d-100, 0, 0]))
    p_prime = np.array(([100, 0, 0], [0, 100, 0], [0, 0, 100]))

    # p = np.array(([d, 200, 100], [d, 100, 0], [d - 100, 100, 100]))
    # p_prime = np.array(([100, 0, 0], [0, 100, 0], [0, 0, 100]))

    # construct intermediate matrix
    Q       = p[1:]       - p[0]
    Q_prime = p_prime[1:] - p_prime[0]

    # calculate rotation matrix
    R = np.dot(np.linalg.inv(np.row_stack((Q, np.cross(*Q)))),
               np.row_stack((Q_prime, np.cross(*Q_prime))))

    # calculate translation vector
    t = p_prime[0] - np.dot(p[0], R)

    # calculate affine transformation matrix
    affine = np.transpose(np.column_stack((np.row_stack((R, t)),
                            (0, 0, 0, 1))))

    extrinsic_matrix = np.around(affine[:3, :], 3)
    # print('ext', extrinsic_matrix)
    return extrinsic_matrix

def normalize_3d(array, scale=1.0, min_scale=0.0):
    min = np.min(array)
    max = np.max(array)
    range = max - min
    print('origin array', np.min(array), np.max(array))
    array = array - min * 1.0
    array = array / range

    array = array * scale
    array = array + min_scale
    print('normalize array', np.min(array), np.max(array))
    return array

@jit
def parallel_ct_to_panel(panel_points, pixel_width, scale):
    for i in range(panel_points.shape[0]):
        panel_points[i][0] = panel_points[i][0] * 1.0 / panel_points[i][2]
        panel_points[i][1] = panel_points[i][1] * 1.0 / panel_points[i][2]

        panel_points[i][0] = pixel_width - int(round(scale * panel_points[i][0]))
        panel_points[i][1] = int(round(scale * panel_points[i][1]))

    return panel_points

@jit
def parallel_panel_to_img(panel, summation_array, voxel_array, pixel_array, img_data, pixel_width, pixel_height):
    panel_len = list(panel.shape)[0]
    for i in range(panel_len):
        xp, yp = panel[i][0], panel[i][1]
        if 0 <= xp < pixel_width and 0 <= yp < pixel_height:
            voxel_location = voxel_array[i]
            xi, yi, zi = int(voxel_location[0]), int(voxel_location[1]), int(voxel_location[2])
            pixel_array[yp][xp] += img_data[xi][yi][zi]
            summation_array[yp][xp] += 1

    return pixel_array, summation_array


def get_non_zero_coord(img_data, voxel_array, world_array):
    print(img_data.shape, voxel_array.shape, world_array.shape)
    len_array = int(list(voxel_array.shape)[0])
    print(len_array)
    non_zero_array = []
    for i in range(len_array):
        if img_data[voxel_array[i][0]][voxel_array[i][1]][voxel_array[i][2]] > 0.0:
            non_zero_array.append(world_array[i])

    return np.array(non_zero_array)


def distance_2_points(p1, p2):
    return math.sqrt((p2[2] - p1[2])**2 + (p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)


def panel_to_coord(pixel, panel_width, panel_height, distance, scale):
    x = -distance * 0.5
    y = -(panel_width * 0.5) + pixel[0] * 1.0 / scale
    z = panel_height * 0.5 - pixel[1] * 1.0 / scale
    return np.array([x,y,z])

def parallel_2D_to_3D(pixel_array, source_location, hull_array, panel_width, panel_height, distance, scale):
    final_3d_coord = []
    len_pixel_array = list(pixel_array.shape)[0]
    len_hull_array = list(hull_array.shape)[0]
    print(len_hull_array)

    for i in range(len_pixel_array):
        pixel = panel_to_coord(pixel_array[i], panel_width, panel_height, distance, scale)
        line_distance = distance_2_points(pixel, source_location)
        print('distance', line_distance, pixel, source_location)
        distance_array = []
        for j in range(len_hull_array):
            panel_distance = distance_2_points(hull_array[j], pixel)
            source_distance = distance_2_points(hull_array[j], source_location)
            distance_array.append([hull_array[j][0], hull_array[j][1], hull_array[j][2], panel_distance + source_distance, source_distance])
        distance_array = np.array(distance_array)
        sorted_array = distance_array[distance_array[:, 3].argsort()]

        check_array = sorted_array[-10:]
        final_point = None

        print(check_array)
        min_distance = np.inf
        for point in check_array:
            if point[3] < 1.1*line_distance and point[4] < min_distance:
                min_distance = point[4]
                final_point = np.array([point[0], point[1], point[2]])

        final_3d_coord.append(final_point)
        print(final_point)

    return np.array(final_3d_coord)


# camera = np.array(([100, 0, 0], [0, 100, 0], [0, 0, 100]))
# world = np.array(([500, 100, 0], [500, 0, -100], [400, 0, 0]))
#
# test_point = np.array([[400, 0, 0, 1], [600, 0, 0, 1]])
# # test_point = np.matmul(T_matrix, np.transpose(test_point))
# affine = calculate_extrinsic_matrix(1000)
# print(affine)
# test_point = np.matmul(affine, np.transpose(test_point))
#
# print('test',np.transpose(test_point))
