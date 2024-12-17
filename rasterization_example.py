import os
import torch
from pytorch3d.io import load_obj
from utils.geom_utils import * 
from utils.file_utils import *
from utils.transformation_utils import *
from rasterizer import Pytorch3dRasterizer
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

# Set device for computation
device = torch.device("cuda:1")
torch.cuda.set_device(device)
image_size = 512

# Define data directory and object file path
DATA_DIR = "./data"
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)
obj_filename = os.path.join(DATA_DIR, "cow_mesh/cow.obj")

# Load 3D object data
verts, faces, aux = load_obj(obj_filename, device=device)
uvcoords = aux.verts_uvs[None, ...]  # Texture coordinates
uvfaces = faces.textures_idx[None, ...]  # Texture faces
verts = verts[None, ...]  # Vertices

# Translate object in space
verts = translate(verts, [0, 0, 10])

# Process faces and texture coordinates
faces = faces.verts_idx[None, ...]
uvcoords = torch.cat([uvcoords, uvcoords[:, :, 0:1] * 0. + 1.], -1)  # Add dummy coordinate for the third UV axis
uvcoords = uvcoords * 2 - 1  # Normalize to [-1, 1]
uvcoords[..., 1] = -uvcoords[..., 1]  # Flip the Y axis to match image space

# Get face UV coordinates and vertex attributes
face_uvcoords = face_vertices(uvcoords, uvfaces)
face_verts = face_vertices(verts, faces)
normals = vertex_normals(verts, faces)
face_normals = face_vertices(normals, faces)

# Combine attributes (UVs, vertex positions, normals)
attributes = torch.cat([face_uvcoords.expand(1, -1, -1, -1), face_verts.detach(), face_normals], dim=-1)

# Initialize rasterizer
rasterizer = Pytorch3dRasterizer()

# Rasterize the object
pix_vals = rasterizer(verts, faces, attributes=attributes)

# Extract different parts of the pixel values
uvcoords_images = pix_vals[:, :3, :, :]
grid = uvcoords_images.permute(0, 2, 3, 1)[:, :, :, :2]  # Reshape grid for texture sampling
verts_image = pix_vals[:, 3:6, :, :]
normal_image = pix_vals[:, 6:9, :, :]
mask = pix_vals[:, -1, :, :][:, None, :, :]  # Create mask image for valid pixels

# Save rasterized images
save_fig(uvcoords_images, f"{RESULTS_DIR}/uvcoords_images")
save_fig(verts_image, f"{RESULTS_DIR}/verts_image")
save_fig(normal_image, f"{RESULTS_DIR}/normal_image")
save_fig(mask, f"{RESULTS_DIR}/mask")

# Load texture image (make sure this path is correct)
image_path = "data/cow_mesh/cow_texture.png"
image = Image.open(image_path)

# Preprocess texture image
transform = transforms.ToTensor()
image_tensor = transform(image)
image_tensor = image_tensor[None, ...]  # Add batch dimension
image_tensor = F.interpolate(image_tensor, [image_size, image_size]).to(device)  # Resize to match rasterizer output

# Sample the texture using the UV coordinates (grid)
albedo_image = F.grid_sample(image_tensor, grid, align_corners=False)

# Save the resulting albedo image
save_fig(albedo_image, f"{RESULTS_DIR}/albedo_image")
