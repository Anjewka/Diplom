from pickle import TRUE
import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
import math

from src import w_z, cwd, OSC_Arr
from visual.read_texture import read_texture

def animation_3d(step, dataSet_2, dataSet_3, dataSet_4, O2):
    #Инициализация окна OpenGL
    pygame.init()
    display = (800, 800)
    pygame.display.set_caption('Satellite')
    screen = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.5, 0.5, 0.5, 1])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 1.0, 1])

    sphere = gluNewQuadric() 

    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0] / display[1]), 100, 800000000.0)

    glMatrixMode(GL_MODELVIEW)
    gluLookAt(0, 40000000, 0, 0, 0, 0, 0, 0, 1)
    viewMatrix = glGetFloatv(GL_MODELVIEW_MATRIX)
    glLoadIdentity()

    iter = 100
    current_step = 1
    texture = read_texture('%s/map.jpg'%(cwd))
    paused = False
    run = TRUE

    displayCenter = [screen.get_size()[i] // 2 for i in range(2)]
    lastPosX = 0
    lastPosY = 0
    while run:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE or event.key == pygame.K_RETURN:
                    run = False
                if event.key == pygame.K_PAUSE or event.key == pygame.K_p:
                    paused = not paused

        if not paused:
            iter += 1
            keypress = pygame.key.get_pressed()
        
            glLoadIdentity()

            #Матрица камеры
            glPushMatrix()
            glLoadIdentity()

            #Движение камеры по нажатию на кнопки
            if keypress[pygame.K_w]:
                glTranslatef(0, 0, 100000)
            if keypress[pygame.K_s]:
                glTranslatef(0, 0, -100000)
            if keypress[pygame.K_d]:
                glTranslatef(-100000, 0, 0)
            if keypress[pygame.K_a]:
                glTranslatef(100000, 0, 0)

            #Вращение камеры с помощью нажатия и перемещения мышки
            if event.type == pygame.MOUSEMOTION:
                x = event.pos[1] - displayCenter[1]
                y = event.pos[0] - displayCenter[0]
                dx = x - lastPosX
                dy = y - lastPosY
                mouseState = pygame.mouse.get_pressed()
                if mouseState[0]:
                    glRotatef(dx * 0.04, 1, 0, 0)
                    glRotatef(dy * 0.04, 0, 1, 0)
                lastPosX = x
                lastPosY = y

            glMultMatrixf(viewMatrix)
            viewMatrix = glGetFloatv(GL_MODELVIEW_MATRIX)

            glPopMatrix()
            glMultMatrixf(viewMatrix)

            glLightfv(GL_LIGHT0, GL_POSITION, [1, -1, 1, 0])

            glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

            glPushMatrix()
            #Шаг
            if iter % step == 0:
                current_step += 1

            #Отрисовка точек массивов
            for it in range(1, min(len(O2), min(current_step, len(dataSet_2[0])))):
                glEnable(GL_POINT_SMOOTH)
                glPointSize(5)

                glBegin(GL_POINTS)

                glColor3d(1, 0, 0)
                #Движение спутника по орбите
                glVertex3d(dataSet_2[0][it], dataSet_2[1][it], dataSet_2[2][it])

                glColor(0.5, 0.5, 0.5)
                #Спутник
                glVertex3d(O2[current_step - 1][0], O2[current_step - 1][1], O2[current_step - 1][2])
                
                glEnd()

                glEnable(GL_POINT_SMOOTH)
                glPointSize(5)
                glBegin(GL_POINTS)
                
                #Движение спутника на земле
                glColor(1.0, 165.0 / 255.0, 0.0)
                glVertex3d(dataSet_3[0][it], dataSet_3[1][it], dataSet_3[2][it])
                glColor(139.0 / 255.0, 0.0, 1.0)
                #Отклонение спутника на угол α
                if(dataSet_4[0][it] and dataSet_4[1][it] and dataSet_4[2][it]):
                    glVertex3d(dataSet_4[0][it], dataSet_4[1][it], dataSet_4[2][it])
                    x_data = dataSet_4[0][it]
                    y_data = dataSet_4[1][it]
                    dataSet_4[0][it] = math.cos(2 * math.pi / 86344) * x_data - math.sin(2 * math.pi / 86344) * y_data
                    dataSet_4[1][it] =  math.sin(2 * math.pi / 86344) * x_data + math.cos(2 * math.pi / 86344) * y_data
                
                #Вращение точек на Земле
                x_data = dataSet_3[0][it]
                y_data = dataSet_3[1][it]
                dataSet_3[0][it] = math.cos(2 * math.pi / 86344) * x_data - math.sin(2 * math.pi / 86344) * y_data
                dataSet_3[1][it] =  math.sin(2 * math.pi / 86344) * x_data + math.cos(2 * math.pi / 86344) * y_data

                glEnd()
                
            if current_step < len(OSC_Arr):
                glBegin(GL_LINES)
                glColor(0.0, 1.0, 0.0)
                glVertex3d(O2[current_step - 1][0] + 1000000 * OSC_Arr[current_step - 1][0][0], O2[current_step - 1][1] + 1000000 * OSC_Arr[current_step - 1][0][1], O2[current_step - 1][2] + 1000000 * OSC_Arr[current_step - 1][0][2])
                glVertex3d(O2[current_step - 1][0], O2[current_step - 1][1], O2[current_step - 1][2])
                glEnd()
                    
                glBegin(GL_LINES)
                glColor(0.0, 1.0, 0.0)
                glVertex3d(O2[current_step - 1][0] + 1000000 * OSC_Arr[current_step - 1][1][0], O2[current_step - 1][1] + 1000000 * OSC_Arr[current_step - 1][1][1], O2[current_step - 1][2] + 1000000 * OSC_Arr[current_step - 1][1][2])
                glVertex3d(O2[current_step - 1][0], O2[current_step - 1][1], O2[current_step - 1][2])
                glEnd()
            
                glBegin(GL_LINES)
                glColor(0.0, 1.0, 0.0)
                glVertex3d(O2[current_step - 1][0] + 1000000 * OSC_Arr[current_step - 1][2][0], O2[current_step - 1][1] + 1000000 * OSC_Arr[current_step - 1][2][1], O2[current_step - 1][2] + 1000000 * OSC_Arr[current_step - 1][2][2])
                glVertex3d(O2[current_step - 1][0], O2[current_step - 1][1], O2[current_step - 1][2])
                glEnd()

            #Матрица Земли
            glPopMatrix()
            glPushMatrix()
        
            #Поворот Земли
            glRotatef(90, 0, 0.0, 1.0)
            glRotatef(180, 0.0, 1.0, 0.0)

            glRotatef(math.degrees(-w_z * iter), 0.0, 0.0, 1.0)

            glEnable(GL_DEPTH_TEST)

            #Загрузка текстуры и Земли
            gluQuadricTexture(sphere, GL_TRUE)
            glEnable(GL_TEXTURE_2D)
            glBindTexture(GL_TEXTURE_2D, texture)
            gluSphere(sphere, 6378245, 100, 100)
            glDisable(GL_TEXTURE_2D)

            glPopMatrix()

            pygame.display.flip()
            pygame.time.wait(5)

    pygame.quit()