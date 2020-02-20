import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import io
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import os

'''
工具函数
继承了我第三次作业的时候写的
四元数求解纯旋转问题中旋转矩阵
'''
def vector2left(y):

	yi = [[0, -y[0], -y[1], -y[2]],
    [y[0], 0, -y[2], y[1]],
    [y[1], y[2], 0, -y[0]],
    [y[2], -y[1], y[0], 0]
    ] 

	return np.array(yi)

def vector2right(x):

	xi = [[   0, -x[0], -x[1], -x[2]], 
	[x[0],     0,  x[2], -x[1]], 
	[x[1], -x[2],     0,  x[0]], 
	[x[2],  x[1], -x[0],    0]
	]

	return np.array(xi)

def q2R(q):
    qw = q[0]
    qv = q[1:4]
    qv = np.reshape(qv, [3, 1])
    qvt = np.reshape(qv,[1, 3]) 

    R = np.dot(float(qw**2 - np.matmul(qvt, qv)), np.identity(3)) + 2 * np.matmul(qv, qvt) + np.dot(2 * qw,  get_qvx(qv))

    return R

def get_qvx(qv):
    
    qvx = [[0, -qv[2], qv[1]],
    [qv[2], 0, -qv[0]],
    [-qv[1], qv[0], 0]]
    
    return qvx

def quaternion_R(X, Y):
    '''
    四元数法求解旋转矩阵
    '''
    A = np.zeros([4,4])

    for i in range(X.shape[1]):
        yi = vector2left(Y[:,i])
        xi = vector2right(X[:,i])

        Ai = np.matmul(yi.T, xi)
        A += Ai 

    _, q_vecs = np.linalg.eig(A)
    
    q = q_vecs[:, 0]

    R = q2R(q)

    return R

def get_T(src, dst):
    
    '''
    使用四元数求解变换矩阵T
    此为一个刚体变换
    '''

    # 分散的对应点中心化
    src_center = np.mean(src, axis=0)
    dst_center = np.mean(dst, axis=0)
    src_centralized = src - src_center
    dst_centralized = dst - dst_center

    # 计算旋转矩阵
    R = quaternion_R(src_centralized.T, dst_centralized.T)

    # 获取t
    t = dst_center.T - np.dot(R, src_center.T)

    # 将R、t整合到单应矩阵中
    T = np.identity(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T

'''
核心部分
'''
def nearest_neighbor(src, dst):
    '''
    给每个src里面的点找dst里面最近的点
    '''
    dist, indexes = NearestNeighbors(n_neighbors=1).fit(dst).kneighbors(src, return_distance=True)
    return dist.flatten(), indexes.flatten()

def icp(A, B, max_iters=100, tolerance=1e-10):

    src = np.ones((4,A.shape[0]))
    dst = np.ones((4,B.shape[0]))
    src[:3,:] = np.copy(A.T)
    dst[:3,:] = np.copy(B.T)

    pre_error = 0
    for i in range(max_iters):
        # 第一步，matching
        dist, indexes = nearest_neighbor(src[:3,:].T, dst[:3,:].T)

        # 第二步，用四元数法求旋转矩阵，作updating
        T = get_T(src[:3,:].T, dst[:3,indexes].T)
        src = np.dot(T, src)

        # 如果收敛了就提前停止
        error = np.sum(dist) / dist.size
        if abs(pre_error-error) < tolerance:
            break
        pre_error = error

    T = get_T(A, src[:3,:].T)

    return T, dist   

if __name__ == '__main__':

    data = io.loadmat('./icp_xy.mat')
    x, y = data['x'], data['y']

    T, dist = icp(x.T, y.T)
    
    src = np.ones((4, x.shape[1]))
    src[:3,:] = np.copy(x)
    y_transformed = np.dot(T, src)[:3,]

    '''
    画图部分
    '''

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x[0,:], x[1,:], x[2,:], c='y', marker='+')
    ax.scatter(y[0,:], y[1,:], y[2,:], c='r', marker='*')
    ax.scatter(y_transformed[0,:], y_transformed[1,:], y_transformed[2,:], c='b', marker='o')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    ax.legend(['source points cloud (src)', 'destination points cloud (dst)', 'src after transformation'])
    plt.show()

    '''
    保存结果
    '''
    if not os.path.exists('./result/'):
        os.makedirs('./result/')    
    io.savemat('./result/solution.mat', {'T':T}) 
