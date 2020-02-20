import numpy as np
import math
from lab3_2 import q2R
from lab3_3_R import R2q
from lab3_3_R import calc_err

def txtSolver(path_x, path_y):
    r'''
    读取txt文件中的点云位置
    '''
    f_x = open(path_x)
    f_y = open(path_y)

    x_lines = f_x.readlines()
    y_lines = f_y.readlines()
    
    # 获取坐标并且减去平均值
    cor_X = np.zeros([len(x_lines), 3], np.float32)
    cor_Y = np.zeros([len(y_lines), 3], np.float32)

    for (i, line) in enumerate(x_lines):
        cor_X[i, :] = line.split()
    cor_X = cor_X - np.mean(cor_X, axis=0)
    
    for (i, line) in enumerate(y_lines):
        cor_Y[i, :] = line.split()
    cor_Y = cor_Y - np.mean(cor_Y, axis=0)

    # 获取真正所需的X、Y矩阵
    X = cor_X
    Y = cor_Y

    return X, Y

def get_X(vector):
    [x1, x2, x3] = vector[:]
    X = [
        [0, -x3, x2],
        [x3, 0, -x1],
        [-x2, x1, 0]
    ]

    return np.asarray(X)

def circle_mul(p, q):
    '''
    完成四元数的圈乘
    '''
    result = np.ones([4,1]).astype(np.float32)
    result[0, 0] = p[0, 0]*q[0, 0] - p[1, 0]*q[1, 0] - p[2, 0]*q[2, 0] - p[3, 0]*q[3, 0]
    result[1, 0] = p[0, 0]*q[1, 0] + p[1, 0]*q[0, 0] + p[2, 0]*q[3, 0] - p[3, 0]*q[2, 0]
    result[2, 0] = p[0, 0]*q[2, 0] - p[1, 0]*q[3, 0] + p[2, 0]*q[0, 0] + p[3, 0]*q[1, 0]
    result[3, 0] = p[0, 0]*q[3, 0] + p[1, 0]*q[2, 0] - p[2, 0]*q[1, 0] + p[3, 0]*q[0, 0]

    return result

'''主函数'''
if __name__ == '__main__':
    path_x = 'rot_x.txt'
    path_y = 'rot_y.txt'
    X, Y = txtSolver(path_x, path_y)

    # 这里设置初始的q
    '''
    R = np.array([
    [-9.97730512e-01, -6.73336586e-02, 5.96927957e-05],
    [ 6.73336834e-02, -9.97730292e-01, 6.63615484e-04],
    [ 1.48736520e-05, 6.66128753e-04, 9.99999778e-01]]
    ).astype(np.float32)
    '''
    # 这里是正常应该解出来的q
    
    q = np.reshape([-3.36859553e-02, -1.86521987e-05, -3.32624853e-04, -9.99432412e-01], [4,1])
    
    #q = np.reshape([1, 0, 0, 0], [4,1])

    ''' pre_q
    pre_q = np.ones_like(q).astype(np.float32)
    '''

    # 求解step部分
    delta_theta = np.zeros([3, 1])
    #pre_delta_theta = np.ones_like(delta_theta)

    J = np.zeros([3*X.shape[0], 3])

    z = np.zeros([3*X.shape[0], 1])
    j = 0

    while(True):
        j += 1
        print('iter ' + str(j) + ' :')
        for i in range(X.shape[0]):
            J[i*3:i*3+3] = -1 * get_X(np.matmul(q2R(np.reshape(q, [4])), np.reshape(X[i], [3,1])))
            z[i*3:i*3+3] = np.reshape(Y[i], [3,1]) - np.matmul(q2R(np.reshape(q, [4])), np.reshape(X[i], [3,1]))
    
        delta_theta = np.matmul(np.matmul(np.linalg.inv(np.matmul(J.T, J)), J.T),z)
        
        exp = np.reshape([1, delta_theta[0, 0]/2, delta_theta[1, 0]/2, delta_theta[2, 0]/2],[4,1])
        
        # 对q进行修正
        q = q / np.linalg.norm(q)

        # 更新
        q = circle_mul(exp, q)
        
        print('the deviation is: ')
        print(calc_err(X,Y,q2R(np.reshape(q, [4]))))
        # 设置迭代停止条件
        if (calc_err(X,Y,q2R(np.reshape(q, [4]))))<1e-4:
            print('iteration stopped!')
            break
        else:
            pre_q = q

        # 对结果进行跟踪
        print('current quaternion is :')
        print(q)
        print('current matrix is :')
        print(q2R(np.reshape(q, [4])))
    
    print('final rotation quaternion is:')
    print(q)
    print('final rotation matrix is:')
    print(q2R(np.reshape(q, [4])))

    