r'''
**********************************
这个py文件实现了作业要求中的：
第十步，Bundle adjustment
**********************************
'''
import numpy as np
import os
import cv2 as cv
from scipy.sparse import lil_matrix
from scipy.optimize import least_squares
from pose_estimation import reprojection_error, get_pose


def pose2proj(poses, intrinsic_matrix):
    '''
    get_pose函数的逆过程
    '''
    projs = []
    for pose in poses:
        rvec = pose[:3]
        tvec = pose[3:]
        R = cv.Rodrigues(rvec)[0]
        t = tvec.reshape((-1,1))
        extrinsic_matrix = np.hstack([R, t])
        projs.append(np.dot(intrinsic_matrix, extrinsic_matrix))

    return projs


class bundle_adjustment:
    def __init__(self, camera_params, points_3d, points_2d, camera_indices, point_indices, intrinsic_matrix):
        '''
        初始化参数
        '''
        self.camera_params = camera_params
        self.points_3d = points_3d
        self.points_2d = points_2d

        self.camera_indices = camera_indices
        self.point_indices = point_indices

        self.intrinsic_matrix = intrinsic_matrix


    def project(self, points, camera_params, intrinsic_matrix):
        '''
        将3D点投影到2D图片中
        '''
        points_proj = np.zeros((points.shape[0], 2), dtype = points.dtype)
        for i, (point, camera_param) in enumerate(zip(points, camera_params), 1):
            points_proj[i - 1] = cv.projectPoints(np.asarray([point]), 
                camera_param[:3].reshape(-1,1), camera_param[3:6].reshape(-1,1), intrinsic_matrix, None)[0]
   
        return points_proj


    def residual(self, params, n_cameras, n_points, camera_indices, point_indices, points_2d, intrinsic_matrix):
        '''
        计算误差
        '''
        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))
        points_proj = self.project(points_3d[point_indices], camera_params[camera_indices], intrinsic_matrix)
        return (points_proj - points_2d).ravel()


    def get_Jacobi(self, n_cameras, n_points, camera_indices, point_indices):
        '''
        得到雅各比矩阵
        '''
        m = camera_indices.size * 2
        n = n_cameras * 6 + n_points * 3
        J = lil_matrix((m, n), dtype=int)

        i = np.arange(camera_indices.size)
        for s in range(6):
            J[2 * i, camera_indices * 6 + s] = 1
            J[2 * i + 1, camera_indices * 6 + s] = 1

        for s in range(3):
            J[2 * i, n_cameras * 6 + point_indices * 3 + s] = 1
            J[2 * i + 1, n_cameras * 6 + point_indices * 3 + s] = 1

        return J


    def extract_params(self, params, n_cameras, n_points):
        '''
        工具函数
        从参数中恢复相机参数和三维点
        '''

        camera_params = params[:n_cameras * 6].reshape((n_cameras, 6))
        points_3d = params[n_cameras * 6:].reshape((n_points, 3))

        return camera_params, points_3d


    def optimize(self):
        '''
        主要优化流程
        '''
        n_cameras = self.camera_params.shape[0]
        n_points = self.points_3d.shape[0]

        x0 = np.hstack((self.camera_params.ravel(), self.points_3d.ravel()))
        f0 = self.residual(x0, n_cameras, n_points, self.camera_indices, self.point_indices, self.points_2d, self.intrinsic_matrix)

        J = self.get_Jacobi(n_cameras, n_points, self.camera_indices, self.point_indices)

        res = least_squares(self.residual, x0, jac_sparsity=J, verbose=2, x_scale='jac', ftol=1e-4, method='trf',
                            args=(n_cameras, n_points, self.camera_indices, self.point_indices, self.points_2d, self.intrinsic_matrix))

        params = self.extract_params(res.x, n_cameras, n_points)

        return params


if __name__ == "__main__":
    from visualization import plot3DReconstruction
    from matplotlib import pyplot as plt

    '''
    将之前问题中的本征矩阵，E矩阵和点云都搬运进来
    '''
    fx = 3303.60037
    fy = 3304.62394
    cx = 1456.64358
    cy = 1946.70050

    intrinsic_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    '''
    路径定义
    '''

    triangulation_path = 'book_pcl'
    keypoints_result = 'book_keypoints'
    camera_pose_result = 'book_camera_pose'
    bundle_adjustment_result = 'book_bundle_adjustment_result'

    '''
    读取点云数据
    '''
    with np.load(os.path.join(keypoints_result, 'inliers_12.npz')) as reader:
        pc12, pc21 = reader['pc1'], reader['pc2']
    with np.load(os.path.join(keypoints_result, 'inliers_13.npz')) as reader:
        pc13, pc31 = reader['pc1'], reader['pc2']
    with np.load(os.path.join(keypoints_result, 'inliers_23.npz')) as reader:
        pc23, pc32 = reader['pc1'], reader['pc2']
    with np.load(os.path.join(triangulation_path, 'reconstrcuted_3d_points.npz')) as reader:
        pc3D= reader['pc']
    with np.load(os.path.join(camera_pose_result, 'camera_projection_matrix.npz')) as reader:
        P1, P2, P3= reader['P2'], reader['P2'], reader['P3']   
    with np.load(os.path.join(camera_pose_result, '3D_2D_points_correspondences.npz')) as reader:
        pc3D, pc1_2D, pc2_2D, pc3_2D = reader['D3'], reader['D2_1'], reader['D2_2'], reader['D2_3']

    '''
    参数准备
    '''
    n_cameras = 3
    n_points = len(pc3D)
    camera_poses = get_pose(np.asarray([[P1], [P2], [P3]]))
    pc2D = np.vstack([pc1_2D, pc2_2D, pc3_2D])
    camera_indices = np.arange(n_cameras).repeat(n_points)
    point_indices = np.tile(np.arange(n_points), n_cameras)

    '''
    核心部分
    '''
    operator = bundle_adjustment(camera_poses, pc3D, pc2D, camera_indices, point_indices, intrinsic_matrix)
    optimal_poses, optimal_pc3D = operator.optimize()

    '''
    获取结果部分
    '''
    optimal_P1, optimal_P2, optimal_P3 = pose2proj(optimal_poses, intrinsic_matrix)

    if not os.path.exists(bundle_adjustment_result):
        os.makedirs(bundle_adjustment_result)
    np.savez(os.path.join(bundle_adjustment_result, 'camera_projection_matrix.npz'), P1=optimal_P1, P2=optimal_P2, P3=optimal_P3)    
    np.savez(os.path.join(bundle_adjustment_result, '3D_2D_points_correspondences.npz'), D3=optimal_pc3D, D2_1=pc1_2D, D2_2=pc2_2D, D2_3=pc3_2D) 

    print('\n\nAfter BA: ')
    camera_poses = get_pose(np.asarray([[optimal_P1], [optimal_P2], [optimal_P3]]), rotation_matrix = True)
    print('Pose 1: \n{}'.format(camera_poses[0]))
    print('Reprojection error 1 : {}\n'.format(reprojection_error(optimal_pc3D, pc1_2D, optimal_P1, None)))
    print('Pose 2 : \n{}'.format(camera_poses[1]))
    print('Reprojection error 2 : {}\n'.format(reprojection_error(optimal_pc3D, pc2_2D, optimal_P2, None)))
    print('Pose 3 : \n{}'.format(camera_poses[2]))
    print('Reprojection error 3 : {}\n'.format(reprojection_error(optimal_pc3D, pc3_2D, optimal_P3, None)))

    '''
    **********************************
    可视化部分
    **********************************
    '''
    colors = [tuple(np.random.randint(0,255,3).tolist()) for i in range(len(optimal_pc3D))]

    
    camera_poses = get_pose(np.asarray([[P1], [P2], [P3]]), rotation_matrix = True)
    plot3DReconstruction(pc3D, colors, camera_poses, figure_name = 'Before BA')
    camera_poses = get_pose(np.asarray([[optimal_P1], [optimal_P2], [optimal_P3]]), rotation_matrix = True)
    plot3DReconstruction(optimal_pc3D, colors, camera_poses, figure_name = 'After BA', output_path = triangulation_path)

    plt.show()
