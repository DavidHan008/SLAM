r'''
**********************************
这个py文件实现了作业要求中的：
第六步，从本质矩阵E中分解出R和t，并选择4个解中正确的一个
第七步，三角化实现三维点云重构
**********************************
'''
import numpy as np
import cv2 as cv
import os

def get_R_t(E):
    '''
    计算四组R和t
    参考了高翔视觉定位十四讲里面的内容
    '''
    U, __, VT = np.linalg.svd(E)

    if np.linalg.det(np.dot(U, VT)) < 0:
        VT = -VT

    '''
    求解出4组可能的解
    '''
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    skew_t = np.dot(np.dot(U, [[0,1,0],[-1,0,0],[0,0,0]]), U.T)
    t = np.array([[-skew_t[1,2],skew_t[0,2],-skew_t[0,1]]]).T

    '''
    存好每一组R和t
    '''
    P2s = [np.hstack((np.dot(U, np.dot(W, VT)), t)),
            np.hstack((np.dot(U, np.dot(W, VT)), -t)),
            np.hstack((np.dot(U, np.dot(W.T, VT)), t)),
            np.hstack((np.dot(U, np.dot(W.T, VT)), -t))]
    return P2s

def one_of_four(P2s, intrinsic_matrix, P1, point1, point2):
    '''
    从四组解中找到正确的一个
    '''
    
    ind = -1
    for i, P2 in enumerate(P2s):
        P2 = np.dot(intrinsic_matrix, P2)
        
        # 对采样的点进行三角化
        d1 = cv.triangulatePoints(P1, P2, point1, point2)
        d1 /= d1[3]

        # camera2world
        P2_homogenous = np.linalg.inv(np.vstack([P2, [0, 0, 0, 1]]))
        d2 = np.dot(P2_homogenous[:3, :4], d1)

        # 同时在两个相机之前，则可以认为是正确的R和t
        if d1[2] > 0 and d2[2] > 0:
            ind = i

    # 返回正确的本征矩阵
    return np.dot(intrinsic_matrix, P2s[ind])


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

    '''
    读取点云数据
    '''
    with np.load(os.path.join(keypoints_result, 'inliers_12.npz')) as reader:
        pc12, pc21 = reader['pc1'], reader['pc2']
    with np.load(os.path.join(keypoints_result, 'E.npz')) as reader:
        E12 = reader['E12']

    
    P1 = np.dot(intrinsic_matrix, np.hstack((np.eye(3), np.zeros((3,1)))))

    # 先计算出4个解
    P2s = get_R_t(E12)
    # 然后根据相机位置的限制，选出一个解
    P2 = one_of_four(P2s, intrinsic_matrix, P1, pc12.T[:,0], pc21.T[:,0])

    # 真正的三角化
    pc3D = cv.triangulatePoints(P1, P2, pc12.T, pc21.T)
    pc3D /= pc3D[3]
    pc3D = pc3D.T[:,:3]

    # 保存结果
    if not os.path.exists(triangulation_path):
        os.makedirs(triangulation_path)
    np.savez(os.path.join(triangulation_path, 'reconstrcuted_3d_points.npz'), pc=pc3D)