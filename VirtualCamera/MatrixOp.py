import numpy as np
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
