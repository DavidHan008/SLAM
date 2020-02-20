r'''
**********************************
这个py文件实现了作业要求中的：
第四步，选取了图1和图2进行特征点的匹配
第五步，使用Ransac估算基本矩阵F与本质矩阵E
**********************************
'''
import cv2 as cv
import numpy as np
import os
import ransac


def fundamentalMat_estimation(pc1, pc2):
    '''
    使用ransac和8点法，估计基础矩阵F
    '''
    F, mask = ransac.my_findF(pc1, pc2, 0)
    mask = mask.ravel()

    # 选择 inlier points
    pc1 = pc1[mask==1]
    pc2 = pc2[mask==1]

    # 带入所有inlier，再算一次
    F, mask = ransac.my_findF(pc1, pc2, 1)
    mask = mask.ravel()

    pc1 = pc1[mask==1]
    pc2 = pc2[mask==1]

    return pc1, pc2, F


def keypoints_matching(img1, img2, output_path = None, inliers_output = 'inliers.npz'):
    '''
    1. 使用了SIFT和SURF特征描述子
    2.使用了FLANN，快速最近邻搜索包
    用于在提取特征描述子之后，进行匹配
    '''
    
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)

    good = []
    pc1 = []
    pc2 = []

    '''
    SIFT部分
    '''
    sift = cv.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    matches = flann.knnMatch(des1, des2, k=2)

    
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance: # 注意这里设置了一个参数0.7，是根据大作业的要求设置的筛选比例
            good.append(m)
            pc1.append(kp1[m.queryIdx].pt)
            pc2.append(kp2[m.trainIdx].pt)

    '''
    SURF部分
    '''
    surf = cv.xfeatures2d.SURF_create()
    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)
    matches = flann.knnMatch(des1, des2, k=2)

    
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            good.append(m)
            pc1.append(kp1[m.queryIdx].pt)
            pc2.append(kp2[m.trainIdx].pt)

    pc1 = np.array(pc1).astype('float64')
    pc2 = np.array(pc2).astype('float64')

    if output_path is not None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        np.savez(os.path.join(output_path, inliers_output), pc1=pc1, pc2=pc2)

    return pc1, pc2

if __name__ == "__main__":
    from visualization import drawEpilines
    from matplotlib import pyplot as plt
    '''
    路径定义
    '''
    source_images = 'book'
    keypoints_result = 'book_keypoints'

    img1 = cv.imread(os.path.join(source_images,'book_01.jpg'))
    img2 = cv.imread(os.path.join(source_images,'book_02.jpg'))
    img3 = cv.imread(os.path.join(source_images,'book_03.jpg'))

    # 将之前问题中的本征矩阵搬运过来
    fx = 3303.60037
    fy = 3304.62394
    cx = 1456.64358
    cy = 1946.70050

    intrinsic_matrix = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])

    print('Matching keypoints ...')

    print('Matching 12..')
    pc12, pc21 = keypoints_matching(img1, img2, 
        output_path = keypoints_result, inliers_output = 'inliers_12.npz')
    pc12, pc21, F12 = fundamentalMat_estimation(pc12, pc21)
    E12 = np.dot(np.dot(intrinsic_matrix.T, F12), intrinsic_matrix)

    print('Matching 13..')
    pc13, pc31 = keypoints_matching(img1, img3, 
        output_path = keypoints_result, inliers_output = 'inliers_13.npz')
    pc13, pc31, F31 = fundamentalMat_estimation(pc13, pc31)
    E31 = np.dot(np.dot(intrinsic_matrix.T, F31), intrinsic_matrix)

    print('Matching 23..')
    pc23, pc32 = keypoints_matching(img2, img3, 
        output_path = keypoints_result, inliers_output = 'inliers_23.npz')
    pc23, pc32, F23 = fundamentalMat_estimation(pc23, pc32)
    E23 = np.dot(np.dot(intrinsic_matrix.T, F23), intrinsic_matrix)

    '''
    将结果保存为E.npz
    后面几步需要使用
    '''
    np.savez(os.path.join(keypoints_result, 'E.npz'), E12=E12, E31=E31, E23=E23)

    '''
    *********************
    可视化部分
    *********************
    '''
    colors = [tuple(np.random.randint(0,255,3).tolist()) for i in range(max(len(pc12), len(pc13), len(pc23)))]
    
    annotated_img12, annotated_img21 = drawEpilines(img1, img2, pc12, pc21, F12, colors)
    
    annotated_img13, annotated_img31 = drawEpilines(img1, img3, pc13, pc31, F31, colors)
    
    annotated_img23, annotated_img32 = drawEpilines(img2, img3, pc23, pc32, F23, colors)

    if not os.path.exists(keypoints_result):
        os.makedirs(keypoints_result)
    cv.imwrite(os.path.join(keypoints_result, 'Epilines_1_2.jpg'), annotated_img12)
    cv.imwrite(os.path.join(keypoints_result, 'Epilines_2_1.jpg'), annotated_img21)
    cv.imwrite(os.path.join(keypoints_result, 'Epilines_1_3.jpg'), annotated_img13)
    cv.imwrite(os.path.join(keypoints_result, 'Epilines_3_1.jpg'), annotated_img31)
    cv.imwrite(os.path.join(keypoints_result, 'Epilines_2_3.jpg'), annotated_img23)
    cv.imwrite(os.path.join(keypoints_result, 'Epilines_3_2.jpg'), annotated_img32)

    plt.show()