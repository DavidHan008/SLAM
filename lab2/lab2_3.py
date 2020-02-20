import numpy as np
import math

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

def txtSolver(path):
    r'''
    读取txt文件中的相机位姿
    '''
    f = open(path)

    lines = f.readlines()

    R = np.zeros([len(lines), 3, 3], np.float32)
    T = np.zeros([len(lines), 3], np.float32)

    for (i, line) in enumerate(lines):
        R[i, :, :] = np.reshape(line.split()[0:9],[3, 3])
        T[i, :] = line.split()[9: 12]

    return R, T


def GLDrawSpaceAxes(AXES_LEN, size):
    r'''
    画出一个坐标轴
    在主函数中使用了两次
    '''
    objCylinder =gluNewQuadric()
    glPushMatrix()
    
    glColor3f(1.0, 1.0, 1.0)
    glutSolidSphere(0.005 / size, 6, 6)
    glColor3f(0.0,0.0, 1.0)
    gluCylinder(objCylinder,0.01 / size, 0.01 / size, AXES_LEN, 10, 5)
    glTranslatef(0,0,AXES_LEN)
    gluCylinder(objCylinder,0.03 / size, 0.0, 0.06 / size, 10, 5)
    glPopMatrix()
    glPushMatrix()
 
    glTranslatef(0,0.5,AXES_LEN)
    glRotatef(90,0.0,1.0,0.0)
    #GLPrint("Z")
    glPopMatrix()
    glPushMatrix()
   
    glColor3f(0.0,1.0, 0.0)
    glRotatef(-90,1.0,0.0,0.0)
    gluCylinder(objCylinder,0.01 / size, 0.01 / size, AXES_LEN, 10, 5)
    glTranslatef(0,0,AXES_LEN)
    gluCylinder(objCylinder,0.03 / size, 0.0, 0.06 / size, 10, 5)
    glPopMatrix()
    glPushMatrix()
    
    glTranslatef(0.5,AXES_LEN,0)
    #GLPrint("Y")
    glPopMatrix()
    glPushMatrix()
    
    glColor3f(1.0,0.0, 0.0)
    glRotatef(90,0.0,1.0,0.0)
    gluCylinder(objCylinder,0.01 / size, 0.01 / size, AXES_LEN, 10, 5)
    glTranslatef(0,0,AXES_LEN)
    gluCylinder(objCylinder,0.03 / size, 0.0, 0.06 / size, 10, 5)
    glPopMatrix()
    glPushMatrix()
    glTranslatef(AXES_LEN,0.5,0)
    #GLPrint("X")
    glPopMatrix()


def solveAngle(R):
    r'''
    根据旋转矩阵，求解对每个轴分别旋转的角度
    '''
    #对应公式中是sqrt(r11*r11+r21*r21)，
    # sy=sqrt(cosβ*cosβ)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6    # 判断β是否为正负90°
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else: 
        x = math.atan2(-R[1, 2], R[1, 1])
        # 当z=0时，此公式也OK
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return x, y, z


if __name__ == "__main__":
    
    # 获取R、T矩阵
    path = './camera_poses.txt'
    R, T = txtSolver(path)
    num_event = R.shape[0]

    # 对T矩阵进行归一化，并且调整尺寸
    # 否则可视化的时候尺寸有点难受
    T = T / np.linalg.norm(T) * 10
    
    # 几个全局变量
    i = 0   # 单位时间
    epoch = 0   # 测试观察使用
    time_scale = 8 # 用于不让它转得那么快

    # points来存储所有轨迹点，画轨迹用
    # 也是一个全局变量
    points = []
    
    def play():
        r'''
        openGL中的callback函数
        更新当前位姿信息
        '''
        global i
        global epoch
        global time_scale
        global points
        i += 1
        #print('pose{}'.format(int(i / time_scale)))
        if (i >= num_event * time_scale):
            i = 0
            points = []
            # 以下供测试使用
            epoch += 1
            print('epoch{}'.format(epoch))
        
        display()
        glutPostRedisplay()

    def init():
        r'''将界面初始化为全黑'''
        glClearColor (0.0, 0.0, 0.0, 0.0)
  
    def display():

        r'''openGL的display函数'''
        
        global points
        glClear (GL_COLOR_BUFFER_BIT)
        
        # 使用MODELVIEW模式
        # 可以使用MODEL_VIEW模式的矩阵计算模式
        glMatrixMode (GL_MODELVIEW)
    
        glLoadIdentity()
        
        #GLDrawSpaceAxes(AXES_LEN)
        r'''设置相机视角'''
        # 归一化向上的方向
        up_direction = (0, 0, 1)
        up_direction = up_direction / np.linalg.norm(up_direction)

        gluLookAt(0.6, 0.3, 1.0,    # 相机所在坐标
        0.0, 0.0, 0.0,  # 目标观察点
        up_direction[0], up_direction[1], up_direction[2])  # 向上的方向

        
        
        # world2body
        # 这个矩阵到底要不要转置存疑。
        # 根据ppt似乎是要转置，但是现在做出来的结果，
        # 不转置才是圆周运动，有点奇怪。
        R_real = R[int(i / time_scale)]
        R_real = R_real.T #/ np.linalg.norm(R_real)
        T1 = - np.matmul(R_real.T, T[int(i / time_scale)])
        

        '''# body2world
        R_real = R[int(i / time_scale)]
        #R_real = R_real #/ np.linalg.norm(R_real)
        T1 = T[int(i / time_scale)]
        '''                     
        
        '''
        #此处为使用全局固定坐标系的尝试
        # 获取GL_MODELVIEW矩阵
        m1 = np.zeros([16])
        m2 = np.zeros([16])
        #旋转    
        m1[0 : 3] = R_real[0, :] / np.linalg.norm(R_real[0])
        m1[4 : 7] = R_real[1, :] / np.linalg.norm(R_real[1])
        m1[8 : 11] = R_real[2, :] / np.linalg.norm(R_real[2])
        m1[15] = 1

        # 平移
        m2[0] = 1
        m2[5] = 1
        m2[10] = 1
        m2[12: 15] = T[i, :] / np.linalg.norm(T[i])
        m2[15] = 1

        
        glMultMatrixf(m2[:])
        glMultMatrixf(m1[:])
        '''
        
        
        
        r'''
        采用局部移动坐标系变换
        '''
        angleX, angleY, angleZ = solveAngle(R_real)
        angleX = angleX / math.pi * 180.0
        angleY = angleY / math.pi * 180.0
        angleZ = angleZ / math.pi * 180.0
        
        #glRotate()
        # 固定不动的世界坐标系
        GLDrawSpaceAxes(AXES_LEN, 10)
        

        # 轨迹生成
        glColor3f (0, 1.0, 0)   #轨迹为绿色
        
        glBegin(GL_LINE_STRIP_ADJACENCY)    # 这里好像，GL_LINE_LOOP也可以，但是会形成一个环，和起点连接上
        for j in range(len(points)):
            glVertex3f(points[j][0], points[j][1], points[j][2])
        glEnd()
        
        r'''
        移动的核心部分
        平移指令在先，旋转指令在后
        实际顺序就是先旋转，再平移
        '''
        glTranslatef(T1[0], T1[1], T1[2])
        points.append(T1)
        
        
        # 对x，y，z分别绕轴旋转
        # 需要按照z,y,x的顺序
        glRotatef(angleZ, 0, 0, 1)
        glRotatef(angleY, 0, 1, 0)
        glRotatef(angleX, 1, 0, 0)
        
        
        # 绘制相机（实际上就是个立方体）
        glColor3f (1.0, 1.0, 1.0)   # 相机是白色的
        glutSolidCube(0.05)

        # 绘制相机坐标系 
        GLDrawSpaceAxes(AXES_LEN/10, 5)
        # glFlush()
        glutSwapBuffers()
    
    
    r'''
    主函数
    '''
    # 窗口初始化等
    glutInit()
    glutInitDisplayMode (GLUT_DOUBLE | GLUT_RGB)
    glutInitWindowSize (1080, 1080)
    glutInitWindowPosition (0, 0)
    glutCreateWindow ("cameraMovement".encode())

    # 控制坐标轴绘制尺寸的超参数
    AXES_LEN = 5

    # 绘图主要区域
    glutDisplayFunc(display)
    glutIdleFunc(play)
    glutMainLoop()
