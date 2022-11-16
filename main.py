import os
from PIL import Image
from utils import *
from rays import *
from tsdf import *


def ProcessDepthImage(file_name, depth_factor):
    """
    Process Depth Image

    Parameters
    ----------
    filename : string
        input depth file
    depth_factor : float
        normalized depth value

    Returns
    -------
    depth_img : ndarray of shape (480, 640)
        filtered depth image
    """
    depth_img = Image.open(file_name).convert('F')
    depth_img = np.array(depth_img) / depth_factor
    scale = np.max(depth_img)
    d_ = depth_img / scale
    d_ = cv2.bilateralFilter(d_, 5, 3, 0.01)
    depth_img = d_ * scale
    return depth_img


def Get3D(depth, K):
    """
        Inverse Projection - create point cloud from depth image

        Parameters
        ----------
        depth : ndarray of shape (H, W)
            filtered depth image
        K : ndarray of shape (3, 3)
            Intrinsic parameters
        Returns
        -------
        point : ndarray of shape (3, H, W)
            Point cloud from depth image
        normal : ndarray of shape (3, H, W)
            Surface normal
    """

    # TODO Your code goes here

    return point, normal


def CreateTSDF(depth, T, voxel_param, K):
    """
        CreateTSDF : VoxelParams class' member function
            Compute distance of each voxel w.r.t a camera

        Parameters
        ----------
        depth : ndarray of shape (H, W)
            Filtered depth image
        T : ndarray of shape (4, 4)
            Transformation that brings camera to world coordinate
        voxel_param : an instance of voxel parameter VoxelParams
        K : ndarray of shape (3, 3)
                Intrinsic parameters
        Returns
        -------
        tsdf : TSDF
            An instance of TSDF with value computed as projective TSDF
    """

    # TODO Your code goes here

    return tsdf


def ComputeTSDFNormal(point, tsdf, voxel_param):
    """
        ComputeTSDFNormal : Compute surface normal from tsdf


        Parameters
        ----------
        point : ndarray of shape (3, H, W)
            Point cloud predicted by casting rays to tsdf
        voxel_param : an instance of voxel parameter VoxelParams
        tsdf : an instance of TSDF

        Returns
        -------
        normal : ndarray of shape (3, H, W)
            Surface normal at each 3D point indicated by 'point' variable

        Note
        -------
        You can use scipy.ndimage.map_coordinates to interpolate ndarray
    """

    # TODO Your code goes here

    return normal


def FindCorrespondence(T, point_pred, normal_pred, point, normal, valid_rays, K, e_p, e_n):
    """
    Find Correspondence between current tsdf and input image's depth/normal

    Parameters
    ----------
    T : ndarray of shape (4, 4)
        Transformation of camera to world coordinate
    point_pred : ndarray of shape (3, H, W)
        Point cloud from ray casting the tsdf
    normal_pred : ndarray of shape (3, H, W)
        Surface normal from tsdf
    point : ndarray of shape (3, H, W)
        Point cloud extracted from depth image
    normal : ndarray of shape (3, H, W)
        Surface normal extracted from depth image
    valid_rays : ndarray of shape (H, W)
        Valid ray casting pixels
    K : ndarray of shape (3, 3)
        Intrinsic parameters
    e_p : float
        Threshold on distance error
    e_n : float
        Threshold on cosine angular error
    Returns
    -------
    Correspondence point of 4 variables
    p_pred, n_pred, p, n : ndarray of shape (3, m)
        Inlier point_pred, normal_pred, point, normal

    """

    # TODO Your code goes here

    return p_pred, n_pred, p, n


def SolveForPose(p_pred, n_pred, p):
    """
        Solve For Incremental Update Pose

        Parameters
        ----------
        p_pred : ndarray of shape (3, -1)
            Inlier tsdf point
        n_pred : ndarray of shape (3, -1)
            Inlier tsdf surface normal
        p : ndarray of shape (3, -1)
            Inlier depth image's point
        Returns
        -------
        deltaT : ndarray of shape (4, 4)
            Incremental updated pose matrix
    """

    # TODO Your code goes here

    return deltaT


def FuseTSDF(tsdf, tsdf_new):
    """
        FuseTSDF : Fusing 2 tsdfs

        Parameters
        ----------
        tsdf, tsdf_new : TSDFs
        Returns
        -------
        tsdf : TSDF
            Fused of tsdf and tsdf_new
    """

    # TODO Your code goes here

    return tsdf


if __name__ == '__main__':
    DEPTH_FOLDER = 'depth_images'
    OUTPUT_FOLDER = 'results'
    if not os.path.exists(OUTPUT_FOLDER):
        os.mkdir(OUTPUT_FOLDER)
    voxel_param = VoxelParams(3, 256)
    fx = 525.0
    fy = 525.0
    cx = 319.5
    cy = 239.5
    K = np.array([[fx, 0., cx], [0., fy, cy], [0., 0., 1]])
    depth_factor = 5000.
    n_iters = 3
    e_p = voxel_param.vox_size * 10.0
    e_n = np.cos(np.pi / 3.0)

    T_cur = np.eye(4)
    depth_file_list = open(os.path.join(DEPTH_FOLDER, 'filelist.list'), 'r').read().split('\n')
    depth_img = ProcessDepthImage(os.path.join(DEPTH_FOLDER, depth_file_list[0]), depth_factor)
    tsdf = CreateTSDF(depth_img, T_cur, voxel_param, K)
    SaveTSDFtoMesh('%s/mesh_initial.ply' % OUTPUT_FOLDER, tsdf)


    rays = ImageRays(K, voxel_param, depth_img.shape)
    for i_frame in range(1, len(depth_file_list)-1):
        print('process frame ', i_frame)

        point_pred, valid_rays = rays.cast(T_cur, voxel_param, tsdf)
        SavePointDepth('%s/pd_%02d.ply' % (OUTPUT_FOLDER, i_frame), point_pred, valid_rays)

        normal_pred = -ComputeTSDFNormal(point_pred, tsdf, voxel_param)
        SavePointNormal('%s/pn_%02d.ply' % (OUTPUT_FOLDER, i_frame), point_pred, normal_pred, valid_rays)

        depth_img = ProcessDepthImage(os.path.join(DEPTH_FOLDER, depth_file_list[i_frame]), depth_factor)
        point, normal = Get3D(depth_img, K)

        for i in range(n_iters):
            p_pred, n_pred, p, n = FindCorrespondence(T_cur, point_pred, normal_pred,
                                                      point, normal, valid_rays, K, e_p, e_n)

            deltaT = SolveForPose(p_pred, n_pred, p)

            # Update pose
            T_cur = deltaT @ T_cur
            u, s, vh = np.linalg.svd(T_cur[:3, :3])
            R = u @ vh
            R *= np.linalg.det(R)
            T_cur[:3, :3] = R


        tsdf_new = CreateTSDF(depth_img, T_cur, voxel_param, K)
        tsdf = FuseTSDF(tsdf, tsdf_new)
        SaveTSDFtoMesh('%s/mesh_%02d.ply' % (OUTPUT_FOLDER, i_frame), tsdf)

    SaveTSDFtoMesh('%s/mesh_final.ply' % OUTPUT_FOLDER, tsdf, viz=True)



