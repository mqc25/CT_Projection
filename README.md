#CT_Projection
-- Minh Cao

##Directory structure
- VirtualCamera: contain all the code that do CT projection
- temp_data (generated after first run of a CT image): hold cache of the necessary arrays to calculation for that particular image
- main.py: example of using the code

##Usage
### Projecting 3D CT to 2D pixel coordinate
- Import the class
```python
from VirtualCamera.VirtualCamera import VirtualCamera
```
- Create an instance
```python
camera = VirtualCamera()
```
- Import CT image
```python
ct_path = 'path\to\nii\file'
camera.import_CT(ct_path)
```
- (Optional) configure panel size, panel distance, pixel resolution and the viewing side. The default value is panel width/height: 640/512 mm, pixel resolution: 1280x1024, distance 1000 mm and view from the left of the head.
```python
camera.configure_panel(panel_height=1024, panel_width=1280, pixel_height=1024, pixel_width=1280, distance=1000, source_side='r'):
```
- add transformation matrix (identity if not being called). This function can be called repeatedly to do series of transformation
```python
rot90z = np.array([[0, 1, 0, 0],
                  [-1, 0, 0, 0],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]])
rot90x = np.array([[1, 0, 0, 0],
                  [0, 0, -1, 0],
                  [0, 1, 0, 0],
                  [0, 0, 0, 1]])
camera.add_transformation_matrix(rot90z)
camera.add_transformation_matrix(rot90x)
```
- do projection
```python
camera.project_3D_to_2D()
```
- show projection image
```python
camera.show_image()
```
- The whole thing should finish in less than 5 second