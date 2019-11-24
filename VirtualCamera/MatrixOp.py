import numpy as np
import math
from numba import jit


@jit
def fill_voxel_array(img_shape, voxel_array):
    for i in range(img_shape[0]):
        for j in range(img_shape[1]):
            for k in range(img_shape[2]):
                voxel_array[i][j][k][0] = i * 1.0
                voxel_array[i][j][k][1] = j * 1.0
                voxel_array[i][j][k][2] = k * 1.0
    return voxel_array

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


@jit
def count_non_zero_element(img_data, shape):
    count = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if img_data[i][j][k] > 0.0:
                    count += 1
    return count


@jit
def get_non_zero_voxel(voxel_array, img_data, shape):
    count = 0
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if img_data[i][j][k] > 0.0:
                    voxel_array[count][0] = i
                    voxel_array[count][1] = j
                    voxel_array[count][2] = k
                    voxel_array[count][3] = 1
                    count += 1


@jit
def distance_2_points(p1, p2):
    distance = math.sqrt((p2[2] - p1[2])**2 + (p2[1] - p1[1])**2 + (p2[0] - p1[0])**2)
    return distance


@jit
def t(p, q, r):
    x = p - q
    return np.dot(r - q, x) / np.dot(x, x)


@jit
def distance_point_line(p, q, r):
    return np.linalg.norm(t(p, q, r)*(p-q)+q-r)


@jit
def panel_to_coord(pixel, panel_width, panel_height, distance, scale):
    x = -distance * 0.5
    y = -(panel_width * 0.5) + pixel[0] * 1.0 / scale
    z = panel_height * 0.5 - pixel[1] * 1.0 / scale
    return np.array([x, y, z])


@jit
def parallel_2D_to_3D(pixel_array, hull_array, hull_array_origin, final_coord, final_coord_original,
                      distance_array, source_location,
                      panel_width, panel_height, distance, scale):
    len_pixel_array = pixel_array.shape[0]
    len_hull_array = hull_array.shape[0]

    for i in range(len_pixel_array):
        pixel_location = panel_to_coord(pixel_array[i], panel_width, panel_height, distance, scale)
        # print('pixel_coord', pixel_location)
        for j in range(len_hull_array):
            source_distance = distance_2_points(hull_array[j], source_location)
            line_distance = distance_point_line(source_location, pixel_location, hull_array[j][:3])
            distance_array[j][0] = j
            distance_array[j][1] = line_distance
            distance_array[j][2] = source_distance

        sorted_array = distance_array[distance_array[:, 1].argsort(kind='mergesort')]
        check_array = sorted_array[:10]
        # print(source_location, pixel_location)

        # print(check_array)
        k = -1
        min_distance = np.inf
        for j in range(10):
            if check_array[j][1] < 1.0 and check_array[j][2] < min_distance:
                min_distance = check_array[j][2]
                k = int(check_array[j][0])

        if k == -1:
            final_coord[i][0] = math.nan
            final_coord[i][1] = math.nan
            final_coord[i][2] = math.nan

            final_coord_original[i][0] = math.nan
            final_coord_original[i][1] = math.nan
            final_coord_original[i][2] = math.nan

            continue

        final_coord[i][0] = hull_array[k][0]
        final_coord[i][1] = hull_array[k][1]
        final_coord[i][2] = hull_array[k][2]

        final_coord_original[i][0] = hull_array_origin[k][0]
        final_coord_original[i][1] = hull_array_origin[k][1]
        final_coord_original[i][2] = hull_array_origin[k][2]

        # print(final_point)

    return final_coord, final_coord_original


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

