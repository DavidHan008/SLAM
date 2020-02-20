r'''
**********************************
这个py文件实现了作业要求中的：
可视化
您可以直接运行这个py文件查看结果，
也可以在分步中，查看其单独的可视化结果
**********************************
'''
import os
import numpy as np
import cv2 as cv
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt


def drawLines(img1, img2, lines, pc1, pc2, colors):

    r,c, __ = img1.shape
    pc1 = np.int32(pc1)
    pc2 = np.int32(pc2)
    for i, (r, pt1, pt2, color) in enumerate(zip(lines, pc1, pc2, colors), 1):
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv.line(img1, (x0,y0), (x1,y1), color, 1)
        img1 = cv.circle(img1, tuple(pt1), 5, color, -1)
        img1 = cv.putText(img1, '{}'.format(i), tuple(pt1), cv.FONT_HERSHEY_SIMPLEX, 1, color, 1, cv.LINE_AA)
        img2 = cv.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2


def drawEpilines(img1, img2, pc1, pc2, F, colors):
    '''
    画对极线，
    用于可视化
    '''

    lines1 = cv.computeCorrespondEpilines(pc2.reshape(-1,1,2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    new_img1, __ = drawLines(img1, img2, lines1, pc1, pc2, colors)

    lines2 = cv.computeCorrespondEpilines(pc1.reshape(-1,1,2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    new_img2, __ = drawLines(img2, img1, lines2, pc2, pc1, colors)

    return new_img1, new_img2


def plot3DReconstruction(pc, colors, poses, figure_name = 'Figure', output_path = None):
    '''
    这个函数借鉴了网上的可视化
    '''

    def scatter3DPoints(pc, colors, ax):

        for i, (pt, c) in enumerate(zip(pc, colors), 1):
            ax.scatter(pt[0], pt[1], pt[2], c = np.asarray([c]) / 255, marker = 'o')
            ax.text(pt[0], pt[1], pt[2], '{}'.format(i), size = 10, zorder = 1, color = np.asarray(c) / 255)
        return ax
            
    def plotCameraPose(pose, idx, ax):

        R, t = pose[:, :3], pose[:,-1]
        pos = np.dot(R, -t).ravel()
        colors = ['r', 'g', 'b']
        for i, c in enumerate(colors):
            ax.quiver(pos[0], pos[1], pos[2], R[0,i], R[1,i], R[2,i], color = c, length = 0.5, normalize = True)
        ax.text(pos[0], pos[1], pos[2], '{}'.format(idx), size = 12, zorder = 1)
        return ax

    fig = plt.figure(figure_name)
    fig.suptitle('3D reconstruction', fontsize = 16)
    ax = fig.gca(projection = '3d')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')

    ax = scatter3DPoints(pc, colors, ax)
    for idx, pose in enumerate(poses, 1):
        ax = plotCameraPose(pose, idx, ax)

    ax.axis('square')
    ax.set_xlim(np.amin(pc[:,0]) - 1)
    ax.set_ylim(np.amin(pc[:,1]) - 1)
    ax.set_zlim(-0.5)
    ax.view_init(-60, -80)

    '''
    最好存矢量图
    '''
    if output_path is not None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        plt.savefig(os.path.join(output_path, '3DPointsCloud.svg'), bbox_inches = 'tight', format = 'svg')    


if __name__ == "__main__":
    from pose_estimation import get_pose
    from keypoint_matching import fundamentalMat_estimation
    '''
    路径定义
    '''
    triangulation_path = 'book_pcl'
    source_images = 'book'
    keypoints_result = 'book_keypoints'
    bundle_adjustment_result = 'book_bundle_adjustment_result'
    camera_pose_result = 'book_camera_pose'

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

    '''
    读取此前的点云数据
    '''
    with np.load(os.path.join(keypoints_result, 'inliers_12.npz')) as reader:
        pc12, pc21 = reader['pc1'], reader['pc2']
    with np.load(os.path.join(keypoints_result, 'inliers_13.npz')) as reader:
        pc13, pc31 = reader['pc1'], reader['pc2']
    with np.load(os.path.join(keypoints_result, 'inliers_23.npz')) as reader:
        pc23, pc32 = reader['pc1'], reader['pc2']
    with np.load(os.path.join(triangulation_path, 'reconstrcuted_3d_points.npz')) as reader:
        pc3D= reader['pc']
    with np.load(os.path.join(camera_pose_result, 'camera_projection_matrix.npz')) as reader:
        P1, P2, P3= reader['P2'], reader['P2'], reader['P3']   
    with np.load(os.path.join(camera_pose_result, '3D_2D_points_correspondences.npz')) as reader:
        pc3D, pc1_2D, pc2_2D, pc3_2D = reader['D3'], reader['D2_1'], reader['D2_2'], reader['D2_3']

    with np.load(os.path.join(bundle_adjustment_result, 'camera_projection_matrix.npz')) as reader:
        optimal_P1, optimal_P2, optimal_P3 = reader['P1'], reader['P2'], reader['P3']
    with np.load(os.path.join(bundle_adjustment_result, '3D_2D_points_correspondences.npz')) as reader:
        optimal_pc3D, pc1_2D, pc2_2D, pc3_2D = reader['D3'], reader['D2_1'], reader['D2_2'], reader['D2_3']

    
    '''
    可视化最核心部分
    '''
    
    colors = [tuple(np.random.randint(0,255,3).tolist()) for i in range(len(optimal_pc3D))]

    
    camera_poses = get_pose(np.asarray([[P1], [P2], [P3]]), rotation_matrix = True)
    plot3DReconstruction(pc3D, colors, camera_poses, figure_name = 'Before BA')
    camera_poses = get_pose(np.asarray([[optimal_P1], [optimal_P2], [optimal_P3]]), rotation_matrix = True)
    plot3DReconstruction(optimal_pc3D, colors, camera_poses, figure_name = 'After BA', output_path = triangulation_path)

    
    __, __, F12 = fundamentalMat_estimation(pc1_2D, pc2_2D)
    annotated_img12, annotated_img21 = drawEpilines(img1, img2, pc1_2D, pc2_2D, F12, colors)
    __, __, F13 = fundamentalMat_estimation(pc1_2D, pc3_2D)
    annotated_img13, annotated_img31 = drawEpilines(img1, img3, pc1_2D, pc3_2D, F13, colors)
    __, __, F23 = fundamentalMat_estimation(pc2_2D, pc3_2D)
    annotated_img23, annotated_img32 = drawEpilines(img2, img3, pc2_2D, pc3_2D, F23, colors)

    if not os.path.exists(keypoints_result):
        os.makedirs(keypoints_result)
    cv.imwrite(os.path.join(keypoints_result, 'Epilines_1_2.jpg'), annotated_img12)
    cv.imwrite(os.path.join(keypoints_result, 'Epilines_2_1.jpg'), annotated_img21)
    cv.imwrite(os.path.join(keypoints_result, 'Epilines_1_3.jpg'), annotated_img13)
    cv.imwrite(os.path.join(keypoints_result, 'Epilines_3_1.jpg'), annotated_img31)
    cv.imwrite(os.path.join(keypoints_result, 'Epilines_2_3.jpg'), annotated_img23)
    cv.imwrite(os.path.join(keypoints_result, 'Epilines_3_2.jpg'), annotated_img32)

    plt.show()