import numpy as np
import nibabel as nib
import h5py
import os
from scipy.signal import convolve2d

import time
from VirtualCamera.MatrixOp import *
from PIL import Image
import PIL.ImageOps

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import math

class VirtualCamera:
    def __init__(self):
        self.img = None
        self.final_img = None
        self.transformation_matrix = None
        self.reverse_matrix = None
        self.img_data = None
        self.pixel_coord = None

        self.img_shape = None
        self.min = None
        self.max = None
        self.range = None

        self.voxel_array = None
        self.world_array = None
        self.color_lookup_table = None

        # print('1', self.img_data[128][128][88])
        # print('min', self.min, 'max', self.max, 'range', self.range, 'mean', np.nanmean(self.img_data))
        # self.img_data, self.voxel_array = self.process_img_data()
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

        self.world = None
        self.source = None
        self.final_coord = None
        self.final_coord_origin = None
        self.pixels = None

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
        self.img_data, self.voxel_array = self.process_img_data()

    def import_hull(self, hull_path):
        self.img = nib.load(hull_path)
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
        self.img_data, self.voxel_array = self.process_img_data()

    def import_pixel_coord(self, pixel_coord):
        self.pixel_coord = pixel_coord

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
        print(self.img_shape)
        voxel_shape = [self.img_shape[0], self.img_shape[1], self.img_shape[2], 4]
        print(voxel_shape)
        voxel_array = np.full(voxel_shape, 1)

        voxel_array = fill_voxel_array(self.img_shape, voxel_array)

        voxel_array = np.reshape(voxel_array, [self.img_shape[0]*self.img_shape[1]*self.img_shape[2], 4])

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

    def convert_voxel_to_world(self, T_matrix=None):
        # print('start converting voxel to CT space')
        # t0 = time.process_time()

        if T_matrix is None:
            world_points = np.transpose(np.matmul(self.transformation_matrix, np.transpose(self.voxel_array)))
        else:
            world_points = np.transpose(np.matmul(T_matrix, np.transpose(self.voxel_array)))

        # print(world_points[128 + 128 * 256 + 88 * 176])
        # print(world_points.shape, self.voxel_array.shape)
        # t1 = time.process_time()
        # print('done converting voxel to CT space', t1 - t0, ' sec')

        return world_points


    def convert_world_to_panel(self, world_points):
        # print('start converting CT space to panel space')
        # t0 = time.process_time()
        panel_points = np.transpose(np.matmul(self.camera_matrix, np.transpose(world_points)))
        panel_points = np.delete(parallel_ct_to_panel(panel_points, self.pixel_width, self.scale), 2, 1)
        # t1 = time.process_time()
        # print('done converting CT space to panel space',  t1 - t0, ' sec')
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

    def project_2D_to_3D(self, pixel):
        self.calculate_camera_matrix()
        count = count_non_zero_element(self.img_data, self.img_shape)
        self.voxel_array = np.empty([count, 4])
        get_non_zero_voxel(self.voxel_array, self.img_data, self.img_shape)
        self.world = self.convert_voxel_to_world()
        world_origin = self.convert_voxel_to_world(self.img.affine)
        self.pixels = pixel
        self.source = np.array([self.distance/2.0, 0, 0])
        self.final_coord = np.full([pixel.shape[0], 3], 999999.0)
        self.final_coord_origin = np.full([pixel.shape[0], 3], 999999.0)

        distance_array = np.full([self.world.shape[0], 3], 99999999.0)
        locations, origin_location = parallel_2D_to_3D(self.pixels, self.world, world_origin, self.final_coord,
                                                       self.final_coord_origin, distance_array, self.source,
                                                       self.panel_width, self.panel_height, self.distance, self.scale)
        return locations, origin_location


    def visualize_2D_to_3D_projection(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.scatter3D(self.source[0], self.source[1], self.source[2], c='k')

        panel_x = np.array([-self.distance/2.0, -self.distance/2.0, -self.distance/2.0, -self.distance/2.0, -self.distance/2.0])
        panel_y = np.array([-self.panel_width/2.0, self.panel_width/2.0, self.panel_width/2.0, -self.panel_width/2.0, -self.panel_width/2.0])
        panel_z = np.array([self.panel_height/2.0, self.panel_height/2.0, -self.panel_height/2.0, -self.panel_height/2.0, self.panel_height/2.0])
        ax.plot3D(panel_x, panel_y, panel_z, 'green')

        for pixel in self.pixels:
            pixel = panel_to_coord(pixel, self.panel_width, self.panel_height, self.distance, self.scale)
            ax.scatter3D(pixel[0], pixel[1], pixel[2], c='k')
            xline = np.array([self.source[0], pixel[0]])
            yline = np.array([self.source[1], pixel[1]])
            zline = np.array([self.source[2], pixel[2]])
            ax.plot3D(xline, yline, zline, 'gray')

        ax.scatter3D(self.world[:, 0], self.world[:, 1], self.world[:, 2], alpha=0.05)

        for i in range(self.final_coord.shape[0]):
            ax.scatter3D(self.final_coord[i][0], self.final_coord[i][1], self.final_coord[i][2], color='r', s=20)
        plt.show()

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