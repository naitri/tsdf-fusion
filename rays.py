import scipy.io
import scipy.ndimage
from tsdf import *
from utils import *


class ImageRays:
    def __init__(self, K, voxel_param=VoxelParams(3, 256), im_size=np.array([480, 640])):
        """
            ImageRays : collection of geometric parameters of rays in an image

            Parameters
            ----------
            K : ndarray of shape (3, 3)
                Intrinsic parameters
            voxel_param : an instance of voxel parameter VoxelParams
            im_size: image size

            Class variables
            -------
            im_size : ndarray of value [H, W]
            rays_d : ndarray of shape (3, H, W)
                Direction of each pixel ray in an image with intrinsic K and size [H, W]
            lambda_step : ndarray (-1, )
                Depth of casted point along each ray direction
        """
        self.im_size = im_size
        h, w = im_size
        xx, yy = np.meshgrid(np.linspace(0, w - 1, w), np.linspace(0, h - 1, h))
        uv1 = np.linalg.inv(K) @ np.reshape(np.concatenate((xx, yy, np.ones_like(xx)), axis=0), (3, h * w))
        self.rays_d = uv1 / np.linalg.norm(uv1, axis=0, keepdims=True)
        self.lambda_step = np.arange(voxel_param.vox_size, voxel_param.phy_len, voxel_param.vox_size)


    def cast(self, T, voxel_param, tsdf):
        """
            cast : ImageRays class' member function
                Collection of geometric parameters of rays in an image

            Parameters
            ----------
            T : ndarray of shape (4, 4)
                Transformation that brings camera to world coordinate
            voxel_param : an instance of voxel parameter VoxelParams
            tsdf : an instance of TSDF

            Returns
            -------
            point_pred : ndarray of shape (3, H, W)
                Point cloud from casting ray to tsdf
            valid : ndarray of shape (H, W)
                Mask to indicate which points are properly casted
        """

        # TODO Your code goes here

        return point_pred, valid

