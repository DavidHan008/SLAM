import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from PIL import Image as Image

def solveV(cor_ori):
    # 通过平行约束求解pure projection变换
    # 使用了scipy库，使用最小二乘法求取数值解
    v = np.ones([2])
    x = np.ones([4])
    y = np.ones([4])
    def func(v):
        # 方程来自我的手动计算
        for i in range(4):
            x[i] = cor_ori[i, 0] / (cor_ori[i, 0] * v[0] + v[1] * cor_tar[i, 1] + 1)
            y[i] = cor_ori[i, 1] / (cor_ori[i, 0] * v[0] + v[1] * cor_tar[i, 1] + 1)

        xfunc1 = x[0] - x[1]
        xfunc2 = x[2] - x[3]
        yfunc1 = y[0] - y[1]
        yfunc2 = y[2] - y[3]

        objectif = [
            xfunc1 - xfunc2,
            yfunc1 - yfunc2
        ]
        return objectif

    v = fsolve(func, [0, 0])
    newCor = np.zeros([4, 2])
    for i in range(4):
        newCor[i, 0] = cor_ori[i, 0] / (cor_ori[i, 0] * v[0] + v[1] * cor_tar[i, 1] + 1)
        newCor[i, 1] = cor_ori[i, 1] / (cor_ori[i, 0] * v[0] + v[1] * cor_tar[i, 1] + 1)

    
    return v, newCor

def solveSR(cor_ori, cor_tar):
    # 使用矩形约束
    # 包括形状约束，和具体尺寸的约束
    A = np.zeros([4])
    x = np.zeros([4])
    y = np.zeros([4])
    def func(A):
        A = np.reshape(A, [2, 2])
        for i in range(4):
            # 注意这里，一般来讲我们认为y是纵坐标
            # 所以把原来的height，即0，对应为y
            x[i] = A[0, 0] * cor_ori[i, 0] + A[0, 1] * cor_ori[i, 1]
            y[i] = A[1, 0] * cor_ori[i, 0] + A[1, 1] * cor_ori[i, 1]
        # 两条边垂直条件
        func1 = np.dot([x[0] - x[1], y[0] - y[1]], [x[1] - x[3], y[1] - y[3]])
        # 平行于世界坐标系条件
        func2 = x[0] - x[1]
        # 尺寸条件，高度和宽度
        func3 = x[2] - x[0] - (cor_tar[2, 0] - cor_tar[0, 0])
        func4 = y[1] - y[0] - (cor_tar[1, 1] - cor_tar[0, 1])
        objectif = [
            func1,
            func2,
            func3,
            func4
        ]
        return objectif

    A = fsolve(func, [0, 0, 0, 0])
    A = np.reshape(A, [2, 2])
    newCor = np.zeros_like(cor_ori)
    for i in range(4):
            # 注意这里，一般来讲我们认为y是纵坐标
            # 所以把原来的height，即0，对应为y
            newCor[i, 0] = A[0, 0] * cor_ori[i, 0] + A[0, 1] * cor_ori[i, 1]
            newCor[i, 1] = A[1, 0] * cor_ori[i, 0] + A[1, 1] * cor_ori[i, 1]

    return A, newCor
    
def solveT(cor_ori, cor_tar):
    # 平移变换，不赘述
    newCor = np.zeros([4, 2])
    tx = np.mean(cor_tar[:, 0] - cor_ori[:, 0])
    ty = np.mean(cor_tar[:, 1] - cor_ori[:, 1])
    T = [tx, ty]
    
    newCor[:, 0] = cor_ori[:, 0] + T[0]
    newCor[:, 1] = cor_ori[:, 1] + T[1]

    return T, newCor

def nonHomo_Cal(cor_ori, cor_tar):
    r'''
    非齐次坐标的解决方案
    使用了scipy库中的最小二乘估计，所以计算出来最终稍有误差
    将projection拆分为三个子部分，分别求解
    '''
    # cor_ori 和 cor_tar 都是4*2的矩阵
    # 先调整成为一个平行四边形， 求出v1，v2，即为两个自由度
    V, parellel = solveV(cor_ori)

    # 第二、三步是对矩形的Affine变换拆分，拆分为rotation&similarity和translate两个部分
    # 第二步，调整为一个矩形
    A, rectangular = solveSR(parellel, cor_tar)

    # 第三步，平移变换
    T, allOK = solveT(rectangular, cor_tar)

    H = np.ones([3, 3])
    H[0:2, 0:2] = A
    H[2, 0:2] = V 
    H[0:2, 2] = T
    H[2, 2] = 1 
    
    return H

def Homogeneous_Cal(cor_ori, cor_tar):
    r'''
    齐次坐标求解
    根据老师课件上的SVD分解方法求解
    '''
    X = np.zeros([4,9])
    Y = np.zeros([4,9])

    # 手算出来的一个结果，直接放上来的
    for i in range(4):
        X[i] = [cor_ori[i, 0], cor_ori[i, 1], 1,
         0, 0, 0,
         -cor_tar[i, 0] * cor_ori[i, 0], -cor_tar[i, 0] * cor_ori[i, 1], -cor_tar[i, 0]]

        Y[i] = [0, 0, 0,
         cor_ori[i, 0], cor_ori[i, 1], 1,
         -cor_tar[i, 1] * cor_ori[i, 0], -cor_tar[i, 1] * cor_ori[i, 1], -cor_tar[i, 1]]
    
    arrays = [X[0], Y[0], X[1], Y[1], X[2], Y[2], X[3], Y[3]]
    A = np.stack(arrays)
    S, V, D = np.linalg.svd(A)
    H = np.reshape(D[8], [3, 3])
    
    H = H / H[2, 2]
    
    return H

def non_Homogeneous_Cal(cor_ori, cor_tar):
    H = np.ones([9])
    r'''
    齐次坐标求解
    根据老师课件上的SVD分解方法求解
    '''
    X = np.zeros([4,8])
    Y = np.zeros([4,8])

    # 手算出来的一个结果，直接放上来的
    for i in range(4):
        X[i] = [cor_ori[i, 0], cor_ori[i, 1], 1,
         0, 0, 0,
         -cor_tar[i, 0]*cor_ori[i, 0], -cor_tar[i, 0]*cor_ori[i, 1]]

        Y[i] = [0, 0, 0,
         cor_ori[i, 0], cor_ori[i, 1], 1,
         -cor_tar[i, 1]*cor_ori[i, 0], -cor_tar[i, 1]*cor_ori[i, 1]]
    
    arrays = [X[0], Y[0], X[1], Y[1], X[2], Y[2], X[3], Y[3]]
    A = np.stack(arrays)
    line = np.array([cor_ori[0, 0], cor_ori[1, 0], 
    cor_ori[1, 0], cor_ori[1, 1],
    cor_ori[2, 0], cor_ori[2, 1],
    cor_ori[3, 0], cor_ori[3, 1]])
    H_line = np.dot(np.linalg.inv(A), line).T
    H[0:8] = H_line
    
    H = np.reshape(H, [3, 3])
    
    return H

def apply_H(source, H):
    r'''
    应用单应变换矩阵
    '''
    result = np.zeros_like(source)
    for i in range(source.shape[0]):
        for j in range(source.shape[1]):
            [new_x, new_y, scale] = np.dot(H_homo, np.array([i, j, 1]).T)
            [new_x, new_y] = [new_x, new_y] / scale
            if new_x < height and new_x >= 0.0 and new_y < width and new_y >= 0:
                result[int(new_x), int(new_y), :] = source[i, j, :]
    return result
    
if __name__ == "__main__":
    
    source_path = './/source//source.jpg'
    Homo_target_path = './/source//Homo_output.jpg'
    nonHomo_target_path = './/source//nonHomo_output.jpg'
    exhibition_path = './/source//exhibition.jpg'
    downsample_rate = 8
    
    source = Image.open(source_path)
    #source = Image.fromarray(source)
    source = np.array(source.resize(
        (int(source.size[0] /downsample_rate),
        int(source.size[1] / downsample_rate)),
        Image.BICUBIC))

    height, width, _ = source.shape

    cor_ori = np.zeros([4, 2])
    cor_tar = np.zeros([4, 2])

    # 利用matplotlib，手动观察四个顶点对应位置
    cor_ori[0] = [25, 120]
    cor_ori[1] = [95, 486]
    cor_ori[2] = [336, 87]
    cor_ori[3] = [338, 469]

    # 使用iphone测量，发现所需矩形高宽比是19：26
    # 0左上 ， 1右上， 2左下， 3右下
    tarHeight, tarWidth = (285, 390)
    cor_tar[0] = [int(height / 2 - tarHeight / 2), int(width / 2 - tarWidth / 2)]
    cor_tar[1] = [int(height / 2 - tarHeight / 2), int(width / 2 + tarWidth / 2)]
    cor_tar[2] = [int(height / 2 + tarHeight / 2), int(width / 2 - tarWidth / 2)]
    cor_tar[3] = [int(height / 2 + tarHeight / 2), int(width / 2 + tarWidth / 2)]

    # 齐次方程解法
    H_homo = Homogeneous_Cal(cor_ori, cor_tar)

    # 带入求解
    result_homo = apply_H(source, H_homo)
    
    # 非齐次解法（解出R、T、V）
    H_nonHomo = non_Homogeneous_Cal(cor_ori, cor_tar)

    # 带入求解
    result_nonHomo = apply_H(source, H_nonHomo)

    # 将结果进行可视化
    fig = plt.figure(figsize=(18, 6))
    gs = plt.GridSpec(1, 3)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    ax1.imshow(source)
    ax1.set_title('Source')
    ax2.imshow(result_homo)
    ax2.set_title('Homogeneous_Way')
    ax3.imshow(result_nonHomo)
    ax3.set_title('nonHomogeneous_Way')

    plt.savefig(exhibition_path)
    plt.show()
    result_homo = Image.fromarray(result_homo)
    result_homo.save(Homo_target_path)
    result_nonHomo = Image.fromarray(result_nonHomo)
    result_nonHomo.save(nonHomo_target_path)

    


