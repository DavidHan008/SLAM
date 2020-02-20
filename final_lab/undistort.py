r'''
**********************************
这个py文件实现了作业要求中的：
第三步，去畸变。
**********************************
'''
import cv2 as cv
import os
import glob
import numpy as np

def undistortion(images_path, intrinsic_matrix, distCoeff, undistorted_images):
    '''
    去畸变
    '''
    if not os.path.exists(undistorted_images):
            os.makedirs(undistorted_images)

    for image_name in glob.glob(os.path.join(images_path, '*.jpg')):

        img = cv.imread(image_name).astype(np.float64)

        h, w = img.shape[:2]
        new_intrinsic_matrix, __ = cv.getOptimalNewCameraMatrix(intrinsic_matrix, distCoeff, (w,h), 1, (w,h))

        # 调用opencv函数作为我们的核心部分
        mapx,mapy = cv.initUndistortRectifyMap(intrinsic_matrix, distCoeff, None, new_intrinsic_matrix, (w,h), 5)
        dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

        
        cv.imwrite(os.path.join(undistorted_images, image_name.split(os.sep)[-1].split('.')[0] + '.jpg'), dst)

if __name__ == "__main__":
    '''
    我的参数来自于Matlab标定工具
    不知道是不是由于相机实际上自带去畸变，
    或者是我的标定棋盘可能已经用了很多年，有了一定变形，
    去畸变效果明显是在加畸变，所以实际上之后我的工作都是使用的没有畸变的图片
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

    distCoeff = np.array([0.10470, -0.39872,  -0.00600, 0.00042, 0.00])

    '''
    路径定义
    '''
    source_images = 'white'
    undistorted_images = 'white_undistorted'

    undistortion(source_images, intrinsic_matrix, distCoeff, undistorted_images)
