from VirtualCamera.VirtualCamera import VirtualCamera
import time
import numpy as np

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

t0 = time.process_time()
ct_path = 'D:\Class\Github\CT_Projection\data\subject_3\hull_subject_3.nii'
camera = VirtualCamera()
camera.import_CT(ct_path)
camera.add_transformation_matrix(rot90x)
camera.add_transformation_matrix(tx)
camera.project_3D_to_2D()
camera.show_image()
t1 = time.process_time()
print(camera.get_above_zero_pixel().shape)
print('time', t1 - t0)
camera.save_img('sample.png')