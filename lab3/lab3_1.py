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
    cor_X = cor_X - np.mean(cor_X,axis=0)
    
    for (i, line) in enumerate(y_lines):
        cor_Y[i, :] = line.split()
    cor_Y = cor_Y - np.mean(cor_Y, axis=0)

    # 获取真正所需的X、Y矩阵
    X = np.zeros([len(x_lines)*3, 9], np.float32)
    Y = np.zeros([len(y_lines)*3, 1], np.float32)

    for (i, line) in enumerate(x_lines):
        X[i * 3, 0:3] = cor_X[i, :]
        X[i * 3 + 1, 3:6] = cor_X[i, :]
        X[i * 3 + 2, 6:9] = cor_X[i, :]
    
    for (i, line) in enumerate(y_lines):
        Y[i*3:i*3+3, 0] = cor_Y[i, :]
    
    return X, Y


if __name__ == '__main__':
    path_x = 'zxl_x.txt'
    path_y = 'zxl_y.txt'
    X, Y = txtSolver(path_x, path_y)

    r = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), Y)
    R = np.reshape(r, [3,3])
    U, sigma, V = np.linalg.svd(R)
    
    sigma = np.identity(3)

    R_regularized = np.dot(np.dot(U, sigma), V)
    R_regularized = R_regularized / np.linalg.det(R_regularized)
    print('rotation matrix is:')
    print(R_regularized)


