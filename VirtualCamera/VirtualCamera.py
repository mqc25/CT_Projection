import numpy as np
import nibabel as nib
import h5py
import os
from scipy.signal import convolve2d

import time
from VirtualCamera.MatrixOp import *
from PIL import Image
import PIL.ImageOps

import math

class VirtualCamera:
    def __init__(self):
        self.img = None
        self.final_img = None
        self.transformation_matrix = None
        self.reverse_matrix = None
        self.img_data = None

        self.img_shape = None
        self.min = None
        self.max = None
        self.range = None

        self.voxel_array = None
        self.world_array = None
        self.color_lookup_table = None

        # print('1', self.img_data[128][128][88])
        # print('min', self.min, 'max', self.max, 'range', self.range, 'mean', np.nanmean(self.img_data))
        self.img_data, self.voxel_array = self.process_img_data()
        # print('2', self.img_data[128][128][88])
        # print('min', np.min(self.img_data), 'max', np.max(self.max), 'range', self.range, 'mean', np.nanmean(self.img_data))
        # im = Image.fromarray(self.img_data[128])
        # im.show()

        self.pixel_array = None
        self.panel_array = None

        self.side = 'r'
        self.panel_height = 1024 / 2.0
        self.panel_width = 1280 / 2.0
        self.pixel_height = int(1024)
        self.pixel_width = int(1280)
        self.distance = 1000
        self.scale = self.pixel_height / self.panel_height

        self.intrinsic_matrix = None
        self.extrinsic_matrix = None
        self.camera_matrix = None


    def import_CT(self, ct_path):
        self.img = nib.load(ct_path)
        self.transformation_matrix = self.img.affine

        self.reverse_matrix = np.linalg.inv(self.transformation_matrix)
        self.img_data = self.img.get_fdata(dtype=np.float32)

        self.img_shape = np.array(self.img_data.shape)
        self.min = np.nanmin(self.img_data)
        if type(self.min) is list:
            self.min = self.min[0]
        self.max = np.nanmax(self.img_data)
        if type(self.max) is list:
            self.max = self.max[0]
        self.range = self.max - self.min

    def process_img_data(self):
        if os.path.exists('./temp_data/preprocessing.h5'):
            print('load file')
            h5f = h5py.File('./temp_data/preprocessing.h5', 'r')

            img_data = h5f['img_data'][:]
            voxel_array = h5f['voxel_array'][:]
            # self.color_lookup_table = h5f['color_table'][:]
            h5f.close()
            return img_data, voxel_array

        # img_data = np.full(self.img_shape, self.min)
        img_data = np.nan_to_num(self.img_data, nan=self.min)
        # img_data -= self.min

        voxel_shape = [self.img_shape[0], self.img_shape[1], self.img_shape[2], 4]
        print(voxel_shape)
        voxel_array = np.full(voxel_shape, 1.0)

        for i in range(self.img_shape[0]):
            for j in range(self.img_shape[1]):
                for k in range(self.img_shape[2]):
                    voxel_array[i][j][k][0] = i
                    voxel_array[i][j][k][1] = j
                    voxel_array[i][j][k][2] = k

        voxel_array = voxel_array.reshape((self.img_shape[0]*self.img_shape[1]*self.img_shape[2], 4))

        if not os.path.exists('./temp_data'):
            os.mkdir('./temp_data')

        h5f = h5py.File('./temp_data/preprocessing.h5', 'w')
        h5f.create_dataset('img_data', data=img_data)
        h5f.create_dataset('voxel_array', data=voxel_array)
        h5f.close()

        return img_data, voxel_array

    def calculate_camera_matrix(self):
        self.intrinsic_matrix = np.array([[self.distance, 0, self.panel_width / 2.0],
                                          [0, self.distance, self.panel_height / 2.0],
                                          [0, 0, 1]])
        self.extrinsic_matrix = calculate_extrinsic_matrix(self.distance, self.side)
        self.camera_matrix = np.matmul(self.intrinsic_matrix, self.extrinsic_matrix)
        # self.camera_matrix = normalize_matrix(self.camera_matrix)

    def configure_panel(self, panel_height=1024, panel_width=1280, pixel_height=1024, pixel_width=1280, distance=1000,
                        source_side='r'):
        self.panel_height = panel_height
        self.panel_width = panel_width
        self.pixel_height = pixel_height
        self.pixel_width = pixel_width
        self.distance = distance
        self.side = source_side

        self.calculate_camera_matrix()

    def add_transformation_matrix(self, T_matrix):
        self.transformation_matrix = np.matmul(T_matrix, self.transformation_matrix)
        self.reverse_matrix = np.linalg.inv(self.transformation_matrix)


    def override_transformation_matrix(self, T_matrix):
        self.transformation_matrix = np.matmul(T_matrix, self.img.affine)
        self.reverse_matrix = np.linalg.inv(self.transformation_matrix)

    def convert_voxel_to_world(self):
        print('start converting voxel to CT space')
        t0 = time.process_time()

        # print(self.voxel_array[128 + 128 * 256 + 88 * 176])
        world_points = np.transpose(np.matmul(self.transformation_matrix, np.transpose(self.voxel_array)))

        # print(world_points[128 + 128 * 256 + 88 * 176])
        # print(world_points.shape, self.voxel_array.shape)
        t1 = time.process_time()
        print('done converting voxel to CT space', t1 - t0, ' sec')

        return world_points


    def convert_world_to_panel(self, world_points):
        print('start converting CT space to panel space')
        t0 = time.process_time()
        panel_points = np.transpose(np.matmul(self.camera_matrix, np.transpose(world_points)))
        panel_points = np.delete(parallel_ct_to_panel(panel_points, self.pixel_width, self.scale), 2, 1)
        t1 = time.process_time()
        print('done converting CT space to panel space',  t1 - t0, ' sec')
        return panel_points.astype(int)

    def save_img(self, name):
        im = Image.fromarray(self.pixel_array)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(name)

    def process_panel_to_img(self, panel):
        print('start summing pixel')
        t0 = time.process_time()

        self.pixel_array = np.full([self.pixel_height, self.pixel_width], 0.0)
        summation_array = np.full([self.pixel_height, self.pixel_width], 0.0)

        self.pixel_array, summation_array = parallel_panel_to_img(panel, summation_array, self.voxel_array,
                                                                  self.pixel_array, self.img_data, self.pixel_width,
                                                                  self.pixel_height)

        # visualize summation array
        # max = np.max(summation_array)
        # print('center', np.where(summation_array == max))
        #
        # print('bias', np.min(summation_array), np.max(summation_array), np.mean(summation_array))
        # v_bias_array = self.normalize_3d(summation_array, self.range, self.min)
        # self.save_img(v_bias_array, 'summation.png')

        print('unalter', np.min(self.pixel_array), np.max(self.pixel_array), np.mean(self.pixel_array))
        self.pixel_array = np.divide(self.pixel_array, summation_array, where=summation_array > 1.0)
        print('alter', np.min(self.pixel_array), np.max(self.pixel_array), np.mean(self.pixel_array))
        # self.pixel_array = normalize_3d(self.pixel_array, self.range, self.min)
        self.pixel_array *= 5000
        self.final_img = self.pixel_array

        t1 = time.process_time()
        print('done summing pixel', t1 - t0, ' sec')

        # self.final_img = self.blur_image(self.pixel_array)

    def project_3D_to_2D(self):
        self.calculate_camera_matrix()
        world = self.convert_voxel_to_world()
        self.panel_array = self.convert_world_to_panel(world)
        print('Process img')

    def show_image(self):
        self.process_panel_to_img(self.panel_array)
        print(self.pixel_array.shape)
        im = Image.fromarray(self.pixel_array)
        im.show()

    def get_above_zero_pixel(self):
        non_zero = []
        for y in range(self.pixel_height):
            for x in range(self.pixel_width):
                if self.final_img[y][x] > 0:
                    non_zero.append([x, y])

        return np.array(non_zero)

    def blur_image(self, array):

        # blur_array = np.full([10,10], 1.0)
        # blur_array /= 100


        # blur_array = np.array([[1, 1, 1, 1, 1],
        #                         [1, 1, 1, 1, 1],
        #                         [1, 1, 2, 1, 1],
        #                         [1, 1, 1, 1, 1],
        #                         [1, 1, 1, 1, 1]])

        # blur_array = np.array([[0.1, 0.1, 0.1],
        #                         [0.1, 0.1, 0.1],
        #                         [0.1, 0.1, 0.1]])

        # blur_array = np.array([[-1, -1, -1],
        #                         [-1, 8, -1],
        #                         [-1, -1, -1]])

        gaussian_blur = np.array([[1, 4, 6, 4, 1],
                                  [4, 16, 24, 16, 4],
                                  [6, 24, 72, 24, 6],
                                  [4, 16, 24, 16, 4],
                                  [1, 4, 6, 4, 1]])
        blur_array = gaussian_blur / 256.0

        # remove cross
        # array = self.remove_cross(array)

        # no processing
        # final_array = self.normalize_3d(array, self.range, self.min)
        # print('final', np.min(final_array), np.max(final_array), np.mean(final_array))


        # array = self.normalize_3d(array)
        final_array = convolve2d(array, blur_array, mode='same')
        final_array = self.normalize_3d(final_array, self.range)
        print('final', np.min(final_array), np.max(final_array), np.mean(final_array))
        im = Image.fromarray(final_array)
        im = im.rotate(-90)
        # im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im.show()
        return im

        # inverted = PIL.ImageOps.invert(im)
        # inverted.show()






# ct_path = 'D:\Class\BE223A_2019\data\subject_3\preopCT_subject_3.nii'
# VirtualCamera(ct_path)