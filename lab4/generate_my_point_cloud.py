import numpy as np
from scipy import io
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os
from icp import icp

if __name__ == "__main__":

    data = io.loadmat('./icp_xy.mat')
    x, y = data['x'], data['y']
    
    '''
    生成随机点云数据
    '''
    my_point_cloud_src = np.random.rand(x.shape[0],x.shape[1])
    #my_point_cloud_dst = np.random.rand(x.shape[0],x.shape[1])

    my_T = np.array([
        [0.91, -0.24478958, 0.33444, 3],
        [0.334915, 0.909699, -0.245517, 2],
        [-0.2441, 0.335447, 0.9098737, 1],
        [0, 0, 0, 1]
    ]).astype(np.float) # 这里可以自定义

    '''
    通过我认为设定的一个变换矩阵，
    求出变换后的点云
    '''
    src = np.ones((4, x.shape[1]))
    src[:3,:] = np.copy(my_point_cloud_src)
    my_point_cloud_dst = np.dot(my_T, src)[:3,]

    '''
    使用icp法尝试求解T矩阵
    '''
    solved_T, dist = icp(my_point_cloud_src.T, my_point_cloud_dst.T)
    
    my_point_cloud_transformed = np.dot(solved_T, src)[:3,]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(my_point_cloud_src[0,:], my_point_cloud_src[1,:], my_point_cloud_src[2,:], c='y', marker='+')
    ax.scatter(my_point_cloud_dst[0,:], my_point_cloud_dst[1,:], my_point_cloud_dst[2,:], c='r', marker='*')
    ax.scatter(my_point_cloud_transformed[0,:], my_point_cloud_transformed[1,:], my_point_cloud_transformed[2,:], c='b', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.legend(['source points cloud (src)', 'destination points cloud (dst)', 'src after transformation'])
    plt.show()

    print('my set T is:')
    print(my_T)
    print('my solved T is:')
    print(solved_T)

