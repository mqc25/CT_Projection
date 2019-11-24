from VirtualCamera.VirtualCamera import VirtualCamera
import time
import numpy as np
from VirtualCamera.MatrixOp import *

rot90z = np.array([[0, 1, 0, 0],
                  [-1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])

rot90x = np.array([[1, 0, 0, 0],
                  [0, 0, -1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]])

tx = np.array([[1, 0, 0, 10],
              [0, 1, 0, 100],
              [0, 0, 1, 10],
              [0, 0, 0, 1]])

identity = np.zeros((4, 4), float)
np.fill_diagonal(identity, 1)
# t0 = time.process_time()
# ct_path = 'D:\Class\Github\CT_Projection\data\subject_3\hull_subject_3.nii'
# camera = VirtualCamera()
# camera.import_CT(ct_path)
# camera.add_transformation_matrix(rot90x)
# camera.add_transformation_matrix(tx)
# camera.project_3D_to_2D()
# camera.show_image()
# t1 = time.process_time()
# print(camera.get_above_zero_pixel().shape)
# print('time', t1 - t0)
# camera.save_img('sample.png')


t0 = time.process_time()
hull_path = 'C:\\Users\\Minh Cao\\Desktop\\Class\\CT_Projection\\data\\subject_3\\hull_subject_3.nii'
camera = VirtualCamera()
camera.import_hull(hull_path)
camera.add_transformation_matrix(identity)
pixel = np.array([[640, 400], [650, 300]])
projected, origin = camera.project_2D_to_3D(pixel)

print(projected)
t1 = time.process_time()
print('time', t1 - t0)
camera.visualize_2D_to_3D_projection()


# p1 = np.array([0.0, 0.0, 0.0])
# p2 = np.array([5.0, 5.0, 5.0])
# p3 = np.array([3.0, 4.0, 3.0])
#
# print(distance_point_line(p1, p2, p3))
