import cv2
import copy
import open3d as o3d
import numpy as np
from skimage import measure
from tsdf import *


def Vec2Skew(x):
    """
    Vec2Skew

    Parameters
    ----------
    x : ndarray of shape (B, 3) or (B, 3, 1)

    Returns
    -------
    Sx : ndarray of shape (B, 3, 3)
        Batch of skew-symmetric matrix
    """
    Sx = None
    if x.ndim == 1: # expect (3,) tensor
        assert x.shape[0] == 3
        B = 1
        Sx = np.zeros((B,3,3))
        Sx[0, 0, 1] = -x[2]
        Sx[0, 1, 0] = x[2]
        Sx[0, 0, 2] = x[1]
        Sx[0, 2, 0] = -x[1]
        Sx[0, 1, 2] = -x[0]
        Sx[0, 2, 1] = x[0]

    elif x.ndim == 2: # expect (B,3) tensor
        assert x.shape[1] == 3
        B = x.shape[0]
        Sx = np.zeros((B, 3, 3))
        Sx[:, 0, 1] = -x[:, 2]
        Sx[:, 1, 0] = x[:, 2]
        Sx[:, 0, 2] = x[:, 1]
        Sx[:, 2, 0] = -x[:, 1]
        Sx[:, 1, 2] = -x[:, 0]
        Sx[:, 2, 1] = x[:, 0]

    elif x.ndim == 3: # expect (B,3,1) tensor
        assert x.shape[1] == 3 and x.shape[2] == 1
        B = x.shape[0]
        Sx = np.zeros((B, 3, 3))
        Sx[:, 0, 1] = -x[:, 2]
        Sx[:, 1, 0] = x[:, 2]
        Sx[:, 0, 2] = x[:, 1]
        Sx[:, 2, 0] = -x[:, 1]
        Sx[:, 1, 2] = -x[:, 0]
        Sx[:, 2, 1] = x[:, 0]

    assert Sx is not None
    return Sx


def Normalize(M, dim=1, tol=1e-6):
    """
    Normalize

    Parameters
    ----------
    M : ndarray of any shape
    dim : int
        Normalized dimension
    tol : float
        Prevent zero division

    Returns
    -------
    M : same as input size
        Normalized input in dimension dim
    """
    M_mag = np.sqrt(np.sum(M**2, axis=dim, keepdims=True)) + tol
    M = M / M_mag
    return M


def SaveTSDFtoMesh(filename, tsdf, viz=False):
    """
    SaveTSDFtoMesh : Draw/Save tsdf as mesh with surface normal as color map

    tsdf: An instant of TSDF
    viz : Enable open3d visualizaton, default False
    """

    verts, faces, _, _ = measure.marching_cubes(tsdf.value, 0, gradient_direction='ascent')
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.fliplr(verts)),
                                     o3d.utility.Vector3iVector(faces))
    mesh.paint_uniform_color([0.4, 0.4, 0.4])
    mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals)
    normals = np.concatenate((normals[:, 1:2], normals[:, 2:3], normals[:, 0:1]), axis=1)
    mesh.vertex_colors = o3d.utility.Vector3dVector(0.5 * (-normals + 1.0))
    if viz:
        o3d.visualization.draw_geometries([mesh])
    o3d.io.write_triangle_mesh(filename, mesh, write_vertex_normals=False, write_vertex_colors=True)



def SavePointDepth(filename, pcd, valid_mask, viz=False):
    """
    SavePointDepth : Draw/Save point cloud with color map jet based on depth

    pcd : ndarray of (3, H, W)
        Point cloud
    valid_mask : ndarray of (H, W)
        Indicate valid points (not drawing invalid points)
    viz : Enable open3d visualizaton, default False
    """
    depth_img = copy.deepcopy(pcd[2])
    depth_img[~valid_mask] = 0
    depth_img[depth_img > 3.0] = 3.0
    depth_img = depth_img / 3.0 * 255
    depth_img = np.array(depth_img, dtype=np.uint8)
    pcd_color = cv2.applyColorMap(depth_img, cv2.COLORMAP_JET)
    pcd_color = pcd_color.reshape(-1, 3).astype(float) / 255.0
    pcd_color = pcd_color[valid_mask.reshape(-1)]

    pcd = pcd.reshape(3, -1)
    pcd = pcd.T
    pcd = pcd[valid_mask.reshape(-1)]
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
    pcd.colors = o3d.utility.Vector3dVector(pcd_color)
    if viz:
        o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(filename, pcd)


def SavePointNormal(filename, pcd, normal, valid_mask, viz=False):
    """
    SavePointNormal : Draw/Save point cloud with surface normal as color map

    pcd : ndarray of (3, H, W)
        Point cloud
    normal : ndarray of (3, H, W)
        Surface normal
    valid_mask : ndarray of (H, W)
        Indicate valid points (not drawing invalid points)
    viz : Enable open3d visualizaton, default False
    """
    pcd = pcd.reshape(3, -1)
    pcd = pcd.T
    pcd = pcd[valid_mask.reshape(-1)]

    normal = normal.reshape(3, -1)
    normal = normal.T
    normal = normal[valid_mask.reshape(-1)]

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd))
    pcd.colors = o3d.utility.Vector3dVector(0.5*(normal+1.0))
    if viz:
        o3d.visualization.draw_geometries([pcd])
    o3d.io.write_point_cloud(filename, pcd)
