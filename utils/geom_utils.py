import numpy as np
import torch
import torch.nn.functional as F

def face_vertices(vertices, faces):
    """ 
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of faces, 3, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    # pytorch only supports long and byte tensors for indexing
    return vertices[faces.long()]

def vertex_normals(vertices, faces):
    """
    :param vertices: [batch size, number of vertices, 3]
    :param faces: [batch size, number of faces, 3]
    :return: [batch size, number of vertices, 3]
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)

    bs, nv = vertices.shape[:2]
    bs, nf = faces.shape[:2]
    device = vertices.device
    vertex_normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None] # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces.long()]

    faces = faces.reshape(-1, 3)
    vertices_faces = vertices_faces.reshape(-1, 3, 3)

    vertex_normals.index_add_(0, faces[:, 1].long(), 
                       torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1]))
    vertex_normals.index_add_(0, faces[:, 2].long(), 
                       torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2]))
    vertex_normals.index_add_(0, faces[:, 0].long(),
                       torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0]))

    vertex_normals = F.normalize(vertex_normals, eps=1e-6, dim=1)
    vertex_normals = vertex_normals.reshape((bs, nv, 3))
    return vertex_normals
