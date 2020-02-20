'''
**********************************
这个py文件实现了Ransac和8点算法
用于实现keypoint_matching.py中的功能
这个py文件没有任何调库
**********************************
'''
import numpy as np

def my_ransac(fit_model, validate_model, X, num_samples, max_iter = -1, thresh = 1.0, ratio_of_inliers = 0.99):
    '''
    自己实现的ransac
    '''
    best_model = None
    best_mask = []
    best_ratio = -1.0

    if max_iter == -1:
        # 注意，这里按照题目要求增加了迭代次数
        max_iter = int(np.log10(1 - ratio_of_inliers) / np.log10(1 - np.power(0.8, 8)) * 10)

    # 迭代开始
    for i in range(max_iter):

        # 随机采样
        all_indices = np.arange(X.shape[0])
        np.random.shuffle(all_indices)
     
        sample_points = X[all_indices[:num_samples],:]
     
        model = fit_model(sample_points)

        if model is None:
            continue
     
        # 计算错误值
        dist = validate_model(model, X)
        mask = np.zeros(len(X)).astype('uint8') 
        mask[np.abs(dist) <= thresh] = 1

        # 保存目前的最佳模型
        if np.count_nonzero(mask) / len(X) > best_ratio:
            best_ratio = np.count_nonzero(mask) / len(X)
            best_model = model
            best_mask = mask
     
        # 终止条件
        if np.count_nonzero(mask) > len(X) * ratio_of_inliers:
            break

    return best_model, best_mask

def my_8P(pc1, pc2):
    '''
    自己实现的8点算法
    '''
    A = np.vstack([
        pc1[:,0]*pc2[:,0], pc1[:,0]*pc2[:,1], pc1[:,0]*pc2[:,2], 
        pc1[:,1]*pc2[:,0], pc1[:,1]*pc2[:,1], pc1[:,1]*pc2[:,2], 
        pc1[:,2]*pc2[:,0], pc1[:,2]*pc2[:,1], pc1[:,2]*pc2[:,2] ]).T

    
    __, __, VT = np.linalg.svd(A)
    
    F = VT[-1].reshape(3,3).T
        
    U, S, VT = np.linalg.svd(F)
    S[-1] = 0
    F = np.dot(np.dot(U, np.diag(S)), VT)
    
    # 注意这里要归一化一下
    return F / F[2,2]


def get_F(points):
    '''
    用8点法，求解基本矩阵
    '''
    pc1 = np.hstack([points[:,:2], np.ones(len(points)).reshape(-1,1)])
    pc2 = np.hstack([points[:,2:], np.ones(len(points)).reshape(-1,1)])

    
    mean1 = np.mean(pc1[:,:2], axis = 0)
    S1 = np.sqrt(2) / np.std(pc1[:,:2], axis = 0).sum()
    T1 = np.array([[S1,0,-S1*mean1[0]], [0,S1,-S1*mean1[1]], [0,0,1]])
    pc1 = np.dot(T1,pc1.T).T
    
    mean2 = np.mean(pc2[:,:2], axis = 0)
    S2 = np.sqrt(2) / np.std(pc2[:,:2], axis = 0).sum()
    T2 = np.array([[S2,0,-S2*mean2[0]], [0,S2,-S2*mean2[1]], [0,0,1]])
    pc2 = np.dot(T2,pc2.T).T

    F = my_8P(pc1, pc2)

    # 反正则化
    return np.dot(np.dot(T2.T, F), T1)


def compute_error(mat, points):
    '''
    作为迭代的指标
    计算整个匹配的误差
    '''
    pc1 = np.hstack([points[:,:2], np.ones(len(points)).reshape(-1,1)])
    pc2 = np.hstack([points[:,2:], np.ones(len(points)).reshape(-1,1)])

    m2 = np.dot(mat, pc1.T)
    s2 = 1 / (m2[0]**2 + m2[1]**2)
    d2 = (pc2.T * m2).sum(axis = 0)
    err2 = d2**2 * s2

    m1 = np.dot(mat.T, pc2.T)
    s1 = 1 / (m1[0]**2 + m1[1]**2)
    d1 = (pc1.T * m1).sum(axis = 0)
    err1 = d1**2 * s1

    return np.where(err1 > err2, err1, err2)


def my_findF(pc1, pc2, method, thresh = 1.0, ratio_of_inliers = 0.99):
    '''
    主函数部分
    作为基本矩阵估计的骨干框架
    '''
    st0 = np.random.get_state()
    # RANSAC和8点法
    if method == 0:
        np.random.seed(0)
        F, mask = my_ransac(get_F, compute_error, 
            np.hstack([pc1, pc2]), num_samples = 8, thresh = thresh, ratio_of_inliers = ratio_of_inliers)
    
    # 8点法
    # 按照题目的意思，再把所有inliers带进来算了一次
    elif method == 1:
        np.random.seed(0)
        F, __ = my_ransac(get_F, compute_error, 
            np.hstack([pc1, pc2]), num_samples = len(pc1), max_iter = 1)   
        mask = np.ones(len(pc1)).astype('uint8')   

    np.random.set_state(st0)
    
    return F, mask