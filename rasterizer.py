import sys
import os
sys.path.append(os.path.abspath(''))
import torch
import torch.nn as nn
from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import rasterize_meshes
from pytorch3d.renderer import RasterizationSettings

# ref https://github.com/yfeng95/DECA/blob/master/decalib/utils/renderer.py
class Pytorch3dRasterizer(nn.Module):
    def __init__(self, image_size=512):
        super().__init__()
        self.raster_settings = RasterizationSettings(
        image_size=image_size, 
        blur_radius=0.0, 
        faces_per_pixel=1, 
        )

    def forward(self, vertices, faces, attributes=None, h=None, w=None):
        fixed_vertices = vertices.clone()
        # fixed_vertices[...,2] = -fixed_vertices[...,2]
        raster_settings = self.raster_settings
        if h is None and w is None:
            image_size = raster_settings.image_size
        else:
            image_size = [h, w]
            if h>w:
                fixed_vertices[..., 1] = fixed_vertices[..., 1]*h/w
            else:
                fixed_vertices[..., 0] = fixed_vertices[..., 0]*w/h
            
        meshes_screen = Meshes(verts=fixed_vertices.float(), faces=faces.long())
        pix_to_face, zbuf, bary_coords, dists = rasterize_meshes(
            meshes_screen,
            image_size=image_size,
            blur_radius=raster_settings.blur_radius,
            faces_per_pixel=raster_settings.faces_per_pixel,
            bin_size=raster_settings.bin_size,
            max_faces_per_bin=raster_settings.max_faces_per_bin,
            perspective_correct=raster_settings.perspective_correct,
        )
        vismask = (pix_to_face > -1).float()
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.clone()
        pix_to_face[mask] = 0
        N, H, W, K, _ = bary_coords.shape

        if attributes is not None:
            D = attributes.shape[-1]
            attributes = attributes.clone()
            attributes = attributes.view(attributes.shape[0]*attributes.shape[1], 3, attributes.shape[-1])
            
            idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
            pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
            pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)

            pixel_vals[mask] = 0
            pixel_vals = pixel_vals[:,:,:,0].permute(0,3,1,2)
            pixel_vals = torch.cat([pixel_vals, vismask[:,:,:,0][:,None,:,:]], dim=1)
            return pixel_vals
        else:
            return vismask[:,:,:,0][:,None,:,:]
    