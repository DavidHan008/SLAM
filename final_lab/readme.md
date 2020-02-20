# SFM

## Intro

SFM, 3D reconstruction with 3 images.

## Steps

0. camera calibration

   using MATLAB calibration tool.

1. undistort.py

   undistortion

2. keypoint_matching.py

   SIFT

   Ransac estimate F and E

3. triangulation.py

   triangulation

4. pose_estimation.py

   3D-2D relations, pnpRansac

5. bundle_adjustment.py

   Bundle adjustment

6. visualization.py

   Results Visualization

7. ransac.py

   Ransac + 8-point algorithm

## requirements

opencv-contrib-python     3.2.0.7

opencv-python             4.1.2.30

numpy                     1.16.5

matplotlib                3.1.1

scipy                     1.3.1
