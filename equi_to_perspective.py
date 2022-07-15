import cv2
from cv2 import VideoWriter_fourcc
import numpy as np
from scipy.spatial.transform import Rotation
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

# Adjust parameters here
output_width = 512
output_height = 512
euler_angles = [0, 2.5*math.pi/4, 0]
focal_length = 0.4*output_width
input_filename = "mediapipe.mov"
output_filename = "output.mp4"
show_steps = False

video = cv2.VideoCapture(input_filename)
ret, input = video.read()
vid_len = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

input_width = input.shape[1]
input_height = input.shape[0]
print("Input video size", input_width, input_height)

# Construct projection, rotation matrices.
camera_matrix = np.array([
    [focal_length, 0, output_width/2],
    [0, focal_length, output_height/2],
    [0, 0, 1]
    ])
inv_camera_matrix = np.linalg.inv(camera_matrix)
rotation = Rotation.from_euler("xyz", euler_angles)
rotation_matrix = rotation.as_matrix()

# Make image coords, a HxWx3x1 array of image space coordinates
x_coords = np.repeat(np.arange(output_width)[None,:,None], output_height, axis=0)
y_coords = np.repeat(output_height - np.arange(output_height)[:,None,None], output_width, axis=1)
z_coords = np.ones([output_height, output_width, 1])
coords = np.concatenate([x_coords, y_coords, z_coords], axis=2)[...,None]

# Inverse project to 3d.
coords_3d = np.matmul(inv_camera_matrix[None,None,...], coords)

# Normalise length to 1
coords_3d_len = np.sqrt(coords_3d[:,:,0,:]**2 + coords_3d[:,:,1,:]**2 + coords_3d[:,:,2,:]**2)
coords_3d_norm = coords_3d / coords_3d_len[:,:,None,:]

# Rotate as desired.
coords_3d_rotated = np.matmul(rotation_matrix[None,None,...], coords_3d_norm)

if show_steps:
    plt.subplot(1,3,1)
    plt.title("x")
    plt.imshow(coords_3d_rotated[:,:,0,0])
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.title("y")
    plt.imshow(coords_3d_rotated[:,:,1,0])
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.title("z")
    plt.imshow(coords_3d_rotated[:,:,2,0])
    plt.colorbar()
    plt.show()

theta = -np.arctan2(coords_3d_rotated[:,:,2,0], coords_3d_rotated[:,:,0,0])
phi = np.arcsin(coords_3d_rotated[:,:,1,0])

equi_location_x = (input_width * (theta + math.pi) / (2*math.pi)).astype(np.float32)
equi_location_y = (input_height * (phi + math.pi/2) / math.pi).astype(np.float32)

if show_steps:
    plt.subplot(2,2,1)
    plt.title("image x")
    plt.imshow(equi_location_x)
    plt.colorbar()
    plt.subplot(2,2,2)
    plt.title("image y")
    plt.imshow(equi_location_y)
    plt.colorbar()
    plt.subplot(2,2,3)
    plt.title("theta/pi")
    plt.imshow(theta / math.pi)
    plt.colorbar()
    plt.subplot(2,2,4)
    plt.title("phi/pi")
    plt.imshow(phi / math.pi)
    plt.colorbar()
    plt.show()

remapped = cv2.remap(input, equi_location_x, equi_location_y, interpolation=cv2.INTER_CUBIC)

if show_steps:
    plt.imshow(remapped)
    plt.show()

output = cv2.VideoWriter(output_filename, VideoWriter_fourcc("m", "p", "4", "v"), 30, remapped.shape[:2])

output.write(remapped)

for i in tqdm(range(vid_len-1)):
    ret, image = video.read()
    if ret:
        remapped = cv2.remap(image, equi_location_x, equi_location_y, interpolation=cv2.INTER_CUBIC)
        output.write(remapped)
output.release()
