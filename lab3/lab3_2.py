import numpy as np

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
    X = np.zeros([len(x_lines), 4], np.float32)
    Y = np.zeros([len(y_lines), 4], np.float32)
    for (i, line) in enumerate(y_lines):
        Y[i, 1:4] = cor_Y[i, :]
        X[i, 1:4] = cor_X[i, :]

    return X, Y

def get_left_quaternion(y):

    #yi = np.zeros([4, 4])
    yi = [[y[0], -y[1], -y[2], -y[3]],
    [y[1], y[0], -y[3], y[2]],
    [y[2], y[3], y[0], -y[1]],
    [y[3], -y[2], y[1], y[0]]
    ]

    return np.array(yi)

def get_right_quaternion(x):

    #xi = np.zeros([4, 4])
    xi = [[x[0], -x[1], -x[2], -x[3]],
    [x[1], x[0], x[3], -x[2]],
    [x[2], -x[3], x[0], x[1]],
    [x[3], x[2], -x[1], x[0]]
    ]

    return np.array(xi)

def get_qvx(qv):
    
    qvx = [[0, -qv[2], qv[1]],
    [qv[2], 0, -qv[0]],
    [-qv[1], qv[0], 0]]
    
    return qvx


def q2R(q):
    qw = q[0]
    qv = q[1:4]
    qv = np.reshape(qv, [3, 1])
    qvt = np.reshape(qv,[1, 3]) 

    R = np.dot(float(qw**2 - np.matmul(qvt, qv)), np.identity(3)) + 2 * np.matmul(qv, qvt) + np.dot(2 * qw,  get_qvx(qv))

    return R

if __name__ == '__main__':
    path_x = 'zxl_x.txt'
    path_y = 'zxl_y.txt'
    X, Y = txtSolver(path_x, path_y)
    A = np.zeros([4, 4])

    for i in range(X.shape[0]):
        yi = get_left_quaternion(Y[i])
        xi = get_right_quaternion(X[i])

        Ai = np.matmul(yi.T, xi)
        A += Ai 

    q_vals, q_vecs = np.linalg.eig(A)
    
    q = q_vecs[:, 0]

    R = q2R(q)

    print('rotation matrix is:')
    print(R)
    print('rotation quaternion is:')
    print(q)