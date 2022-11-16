import numpy as np

class VoxelParams:
    def __init__(self, phy_len, num_x):
        """
            VoxelParams : voxel representation class

            Parameters
            ----------
            phy_len: physical length of the voxel
            num_x: number of discrete point representing the physical space

            Class variables
            -------
            voxel_x, voxel_y, voxel_z  : ndarray(H, W, D)
                Voxel's physical position w.r.t the world
        """
        self.num_x = num_x
        self.phy_len = phy_len
        self.phy_w = phy_len
        self.phy_d = phy_len
        self.volume = num_x ** 3
        self.vox_size = phy_len / num_x
        self.trunc_thr = self.vox_size * 3
        self.trunc_thr_inv = 1.0 / self.trunc_thr
        self.voxel_origin = np.array([-self.phy_len / 2.0, -self.phy_len / 2.0, self.phy_len / 8.0])

        x, y, z = np.meshgrid(np.linspace(0, self.num_x - 1, self.num_x),
                              np.linspace(0, self.num_x - 1, self.num_x),
                              np.linspace(0, self.num_x - 1, self.num_x))
        # Physical location of each voxel
        self.voxel_x = self.vox_size * x + self.voxel_origin[0]
        self.voxel_y = self.vox_size * y + self.voxel_origin[1]
        self.voxel_z = self.vox_size * z + self.voxel_origin[2]


class TSDF:
    def __init__(self, voxel_param=VoxelParams(3, 256), sdf=None, valid_voxel=None):
        """
            TSDF : Truncated Signed Distance Function class

            Parameters
            ----------
            voxel_param : an instance of voxel parameter VoxelParams
            sdf: ndarray of (H, W, D) - same shape as voxel dimension in voxel_param
                Contains signed distance function of each voxel
            valid_voxel: ndarray of (H, W, D) - same shape as voxel dimension in voxel_param
                Indicates which voxel's values are valid

            Class variables
            -------
            value: ndarray of (H, W, D)
                Truncated signed distance value of each voxel. Used for casting rays to TSDF
            weight: ndarray of (H, W, D)
                Used for the fusion of multiple TSDF
        """
        self.value = sdf
        self.value[self.value < -voxel_param.trunc_thr] = voxel_param.trunc_thr
        self.value[self.value > voxel_param.trunc_thr] = voxel_param.trunc_thr
        self.value = self.value / voxel_param.trunc_thr
        self.weight = np.zeros((voxel_param.num_x, voxel_param.num_x, voxel_param.num_x))
        self.weight[np.abs(sdf) < voxel_param.trunc_thr] = 1
        self.voxel_x = voxel_param.voxel_x
        self.voxel_y = voxel_param.voxel_y
        self.voxel_z = voxel_param.voxel_z
        self.valid = valid_voxel
        self.weight[~valid_voxel] = 0.0