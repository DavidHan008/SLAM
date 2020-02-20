r'''
**********************************
这个py文件实现了作业要求中的：
第八步，对1、3和2、3进行特征匹配，得到第三个视角的3D-2D对应关系
第九步，使用pnpRansac实现第三个视角的相机姿态估算
**********************************
'''
import os
import numpy as np
import cv2 as cv
from triangulation import get_R_t, one_of_four

def reprojection_error(objpoints, imgpoints, projection_matrix, distortion_vector):
    '''
    计算重投影误差
    非常重要，是我们的目标函数
    '''
    intrinsic_matrix = cv.decomposeProjectionMatrix(projection_matrix)[0]
    # 计算[R|t]
    extrinsic_matrix = np.dot(np.linalg.inv(intrinsic_matrix), projection_matrix)
    R, t = extrinsic_matrix[:3, :3], extrinsic_matrix[:, 3]

    
    imgpoints2 = cv.projectPoints(objpoints, cv.Rodrigues(R)[0], t, intrinsic_matrix, distortion_vector)[0]
    
    imgpoints1 = np.asarray([[row.tolist()] for row in imgpoints])
    error = cv.norm(imgpoints1, imgpoints2, cv.NORM_L2) / len(imgpoints2)

    return error

def get_pose(projs, rotation_matrix = False):
    '''
    从投影矩阵中提取相机位姿
    以及，是否需要保留R和t的形式，
    即Extrinsic的形式
    '''
    poses = []
    for proj in projs:
        intrinsic_matrix = cv.decomposeProjectionMatrix(proj[0])[0]
        extrinsic_matrix = np.dot(np.linalg.inv(intrinsic_matrix), proj[0])
        R, t = extrinsic_matrix[:3, :3], extrinsic_matrix[:, 3]
        if rotation_matrix:
            poses.append(np.hstack([R, t.reshape(-1,1)]))
        else:
            '''
            bundle adjustment会使用这个部分的计算
            '''
            rvec = cv.Rodrigues(R)[0].ravel()
            tvec = t.ravel()
            poses.append(np.hstack([rvec, tvec]))

    return np.asarray(poses)

def get_3d2d_relations(pc1, pc2, pc3D, pc_ref):
    '''
    找到3D-2D关系
    '''
    mask = same_keypoints(pc_ref, pc1)
    
    map_2Dto3D = {idx2D - 1:idx3D for idx2D, idx3D in enumerate(mask, 1) if idx3D != -1}
    return pc2[list(map_2Dto3D.keys())], pc3D[list(map_2Dto3D.values())]

def same_keypoints(pc1, pc2):
    '''
    找到相同关键点
    '''
    mask = -np.ones((len(pc2))).astype('int')
    for i, row in enumerate(pc2, 1):
        idx = np.where((np.abs(pc1 - row) < 1e-2).all(1))[0]
        if idx.size:
            mask[i - 1] = idx[0]
    return mask

if __name__ == "__main__":
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
    with np.load(os.path.join(keypoints_result, 'E.npz')) as reader:
        E12 = reader['E12']
    
    '''
    第八步的核心
    '''
    print('Matching 13..')
    pc3_2D_1, pc3_3D_1 = get_3d2d_relations(pc13, pc31, pc3D, pc12)
    print('Find {} common matches.'.format(len(pc3_2D_1)))

    print('Matching 23..')
    pc3_2D_2, pc3_3D_2 = get_3d2d_relations(pc23, pc32, pc3D, pc21)
    print('Find {} common matches.'.format(len(pc3_2D_2)))

    
    pc3_2D = pc3_2D_1
    pc3_3D = pc3_3D_1
    for i, row in enumerate(pc3_2D_2, 1):
        # 取交集
        idx = np.where((pc3_2D == row).all(1))[0]
        if idx.size:
            continue
        pc3_2D = np.vstack([pc3_2D, row])
        pc3_3D = np.vstack([pc3_3D, pc3_3D_2[i-1]])

    '''
    第九步的核心
    直接调用opencv库了
    '''
    __, rvec, t3, __ = cv.solvePnPRansac(pc3_3D, pc3_2D, intrinsic_matrix, None)
    
    '''
    这些步骤来自三角化
    '''
    P1 = np.dot(intrinsic_matrix, np.hstack((np.eye(3), np.zeros((3,1)))))
    P2s = get_R_t(E12)
    P2 = one_of_four(P2s, intrinsic_matrix, P1, pc12.T[:,0], pc21.T[:,0])
    P3 = np.dot(intrinsic_matrix, np.hstack((cv.Rodrigues(rvec)[0], t3)))

    # 更新第一视角和第二视角的2D-3D对应关系
    pc1_2D = []
    pc2_2D = []
    for pt in pc3_3D:
        idx = np.where((pc3D == pt).all(1))[0]
        pc1_2D.append(pc12[idx][0])
        pc2_2D.append(pc21[idx][0])
    pc1_2D = np.asarray(pc1_2D)
    pc2_2D = np.asarray(pc2_2D)
    pc3D = pc3_3D

    if not os.path.exists(camera_pose_result):
        os.makedirs(camera_pose_result)
    np.savez(os.path.join(camera_pose_result, 'camera_projection_matrix.npz'), P1=P1, P2=P2, P3=P3)    
    np.savez(os.path.join(camera_pose_result, '3D_2D_points_correspondences.npz'), D3=pc3D, D2_1=pc1_2D, D2_2=pc2_2D, D2_3=pc3_2D) 

    print('\n\nPose estimation results: ')
    camera_poses = get_pose(np.asarray([[P1], [P2], [P3]]), rotation_matrix = True)
    print('Pose (extrinsic matrix) of the first view : \n{}'.format(camera_poses[0]))
    print('Reprojection error on the first view : {}\n'.format(reprojection_error(pc3D, pc1_2D, P1, None)))
    print('Pose (extrinsic matrix) of the second view : \n{}'.format(camera_poses[1]))
    print('Reprojection error on the second view : {}\n'.format(reprojection_error(pc3D, pc2_2D, P2, None)))
    print('Pose (extrinsic matrix) of the third view : \n{}'.format(camera_poses[2]))
    print('Reprojection error on the third view : {}\n'.format(reprojection_error(pc3D, pc3_2D, P3, None)))
