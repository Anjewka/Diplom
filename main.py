import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.cbook as cbook
from src import *

from basic.calc_grinvich_coords import calc_grinvich_coords
from basic.dots_array import dots_array
from basic.calc_KSK import calc_KSK

from visual.animation_2d import map_2d
from visual.animation_3d import animation_3d
from visual.trace_points import trace_points

from speed_field.conversion import conversion
from speed_field.calculate_earth_point import calculate_earth_point
from speed_field.calculate_result_point import calculate_result_point

from min_field.calc_min_field import calc_min_field
from min_field.speed_vector import speed_vector
from min_field.speed_vector import maxDV

from half_plate.fsk import fsk
from half_plate.fsk_dots import fsk_dots
from half_plate.earth_dots_picture import earth_dots_picture
from half_plate.deltime import deltime

from plate import Plate, Plates

from pickle import TRUE
import pygame
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from PIL import Image
import math

import pandas as pd
        
def calculate(mode, coords_plane2, angels): 
    for t_1 in np.arange(t_0 - step, t + step, step):
        [x_rv, y_rv, z_rv, r, OSC, O1, e3] = calc_grinvich_coords(t_1, a, e, i, t_0, step, d)
        K1, KSC = calc_KSK(angels[0], angels[1], angels[2], O1, OSC, r, d)
        dots_array(x_rv, y_rv, z_rv, r, O1, OSC, K1, KSC)
    if mode != 'VISUAL':
        # print(K1, KSC)
        calculate_earth_point(O1, K1, KSC, coords_plane2, e_point_x, e_point_y, e_point_z, delta_t)
        [x_rv, y_rv, z_rv, r, OSC, O1, e3] = calc_grinvich_coords(t_1 + delta_t, a, e, i, t_0, step, d)
        K1_new, KSC_new = calc_KSK(angels[0], angels[1], angels[2], O1, OSC, r, d)
        calculate_result_point(O1, K1_new, KSC_new, e3, coords_plane2, e_point_x, e_point_y, e_point_z)
        
def calculate_2(mode, coords_plane2):
    for t_1 in np.arange(t_0 - step, t + step, step):
        [x_rv, y_rv, z_rv, r, OSC, O1, e3] = calc_grinvich_coords(t_1, a, e, i, t_0, step, d)

    if mode == 'HALF_PLATE':
        K1, KSC = calc_KSK(0, 0, 0, O1, OSC, r, d)
        # print(np.allclose(np.eye(3), np.dot(OSC.T, OSC))) # проверяем ортогональность
        # print(np.allclose(np.eye(3), np.dot(KSC.T, KSC))) # проверяем ортогональность
        #находим центр и направляющие векторы ФСК -  фото поверхности земли
        fsk_center, FSK, fsk_x = fsk(O1, K1, KSC, a0, b0 + m1/2)
        t_2 = deltime(fsk_center, fsk_x)
    else:
        K1_nadir, KSC_nadir = calc_KSK(0, 0, 0, O1, OSC, r, d)
        #первый угол - тангаж, поменяй его на любой, второй оставь ноль, рыскание не нужно
        pitch_ang = math.pi/90
        roll_ang = math.pi/60
        yaw_ang = 0
        K1, KSC = calc_KSK(pitch_ang, roll_ang, yaw_ang, O1, OSC, r, d)
        #находим центр и направляющие векторы ФСК -  фото поверхности земли
        fsk_center, FSK, fsk_x = fsk(O1, K1_nadir, KSC_nadir, a0, b0 + m1/2)
        t_2 = deltime(fsk_center, fsk_x)

    #находим время, спустя которое нужно сделать следующий кадр
    #количество снимков
    k = 5
    e_point_x = [[] for _ in range(k)]
    e_point_y = [[] for _ in range(k)]
    e_point_z = [[] for _ in range(k)]
    calculate_earth_point(O1, K1, KSC, coords_plane2, e_point_x[0], e_point_y[0], e_point_z[0], (k-1)*t_2)
    earth_dots.append(fsk_dots(fsk_center, FSK, e_point_x[0], e_point_y[0], e_point_z[0], O1))

    for it in range(1, k):
        [x_rv, y_rv, z_rv, r, OSC, O1, e3] = calc_grinvich_coords(t_1 + it*t_2, a, e, i, t_0, step, d)
        if mode == 'HALF_PLATE':
            K1_new, KSC_new = calc_KSK(0, 0, 0, O1, OSC, r, d)
        else:
            K1_new, KSC_new = calc_KSK(pitch_ang, roll_ang, yaw_ang, O1, OSC, r, d)#it*-math.pi/1200)
        calculate_earth_point(O1, K1_new, KSC_new, coords_plane2, e_point_x[it], e_point_y[it], e_point_z[it], (k-it-1)*t_2)
        earth_dots.append(fsk_dots(fsk_center, FSK,  e_point_x[it], e_point_y[it], e_point_z[it], O1))

    earth_dots_picture(earth_dots, fsk_center, fsk_x, FSK, t_2, n0)

def start():
    coords_x.clear()
    coords_y.clear()
    V_x_alpha.clear()
    V_y_alpha.clear()

    V_x_new_no_alpha.clear()
    V_y_new_no_alpha.clear()

    V_x_new.clear()
    V_y_new.clear()

    V_x_old.clear()
    V_x_old.clear()

    V_x.clear()
    V_y.clear()
  
def calculate_3(coords_plane2, angels, k, nums, deg, prev_a):
    # Положение спутника во времени в нулевой момент
    arra = []

    for t_1 in np.arange(t_0 - step, t + step, step):
        [x_rv, y_rv, z_rv, r, OSC, O1, e3] = calc_grinvich_coords(t_1, a, e, i, t_0, step, d)
    
    # Направление внадир
    K1_nadir, KSC_nadir = calc_KSK(0, 0, 0, O1, OSC, r, d)
    # Первый угол - тангаж, поменяй его на любой, второй оставь ноль, рыскание не нужно
    # Направление задается самолетными углами
    K1, KSC = calc_KSK(angels[0], angels[1], angels[2], O1, OSC, r, d)
    # Находим центр и направляющие векторы ФСК -  фото поверхности земли
    fsk_center, FSK, fsk_x = fsk(O1, K1_nadir, KSC_nadir, a0, b0 + m1 / 2)
    t_2 = deltime(fsk_center, fsk_x)

    # Сдвинуть спутник на расстояние предыдущего слоя
    if nums > 0:
        [x_rv, y_rv, z_rv, r, OSC, O1, e3] = calc_grinvich_coords(t_1 + nums * t_2, a, e, i, t_0, step, d)
        K1, KSC = calc_KSK(angels[0], angels[1], angels[2], O1, OSC, r, d)

    # Координаты в ФСК
    e_point_x = [[] for _ in range(k)]
    e_point_y = [[] for _ in range(k)]
    e_point_z = [[] for _ in range(k)]
    
    # Координаты пластины на Земле в ИСК
    ep_x = []
    ep_y = []
    ep_z = []
    
    # Для полей скоростей
    coords = [[0] * m0 for i in range(n0)]

    # Вычислить координаты пластины на Земле
    calculate_earth_point(O1, K1, KSC, coords_plane2, e_point_x[0], e_point_y[0], e_point_z[0], (k - 1) * t_2)
    earth_dots.append(fsk_dots(fsk_center, FSK, e_point_x[0], e_point_y[0], e_point_z[0], O1))
    
    # Вычисляем координаты точек матрицы на Земле
    calculate_earth_point(O1, K1, KSC, coords_plane2, ep_x, ep_y, ep_z, delta_t)
    
    # Сдвигаем спутник на время экспозиции
    if nums > 0:
        [x_rv1, y_rv1, z_rv1, r1, OSC1, O11, e31] = calc_grinvich_coords(t_1 + nums * t_2 + delta_t, a, e, i, t_0, step, d)
        K1_new, KSC_new = calc_KSK(angels[0], angels[1], angels[2], O11, OSC1, r1, d)
    else:
        [x_rv1, y_rv1, z_rv1, r1, OSC1, O11, e31] = calc_grinvich_coords(t_1 + delta_t, a, e, i, t_0, step, d)
    K1_new, KSC_new = calc_KSK(angels[0], angels[1], angels[2], O11, OSC1, r1, d)
    # Точки матрицы в КСК
    calculate_result_point(O11, K1_new, KSC_new, e3, coords_plane2, ep_x, ep_y, ep_z)
    V_x_2 = [[0] * n0 for i in range(n0)]
    V_y_2 = [[0] * m0 for i in range(n0)]

    [phi, u_x, u_y] = calc_min_field(a0, b0, n0, m0, l0, coords_plane2, coords, V_x, V_y, V_x_2, V_y_2)
    arra.append(maxDV(V_x, V_y, u_x, u_y, coords_plane2))
    # Очистка массивов, использовавшихся для расчета полей скоростей
    start()

    for it in range(1, k - nums):
        # Для полей скоростей
        coords = [[0] * m0 for i in range(n0)]
        V_x_2 = [[0] * n0 for i in range(n0)]
        V_y_2 = [[0] * m0 for i in range(n0)]

        # Координаты пластины на Земле в ИСК
        ep_x = []
        ep_y = []
        ep_z = []

        # Сдвигаем спутник в нужное место
        [x_rv, y_rv, z_rv, r, OSC, O1, e3] = calc_grinvich_coords(t_1 + (it + nums) * t_2, a, e, i, t_0, step, d)
        # Определяем СК с помощью самолетных углов
        K1, KSC = calc_KSK(angels[0], angels[1], angels[2], O1, OSC, r, d)        
        # Вычисляем координаты точек матрицы на Земле
        calculate_earth_point(O1, K1, KSC, coords_plane2, ep_x, ep_y, ep_z, delta_t)
        
        # Сдвигаем спутник на время экспозиции
        [x_rv1, y_rv1, z_rv1, r1, OSC1, O11, e31] = calc_grinvich_coords(t_1 + (it + nums) * t_2 + delta_t, a, e, i, t_0, step, d)
        K1_new, KSC_new = calc_KSK(angels[0], angels[1], angels[2], O11, OSC1, r1, d)
        
        # Точки матрицы в КСК
        calculate_result_point(O11, K1_new, KSC_new, e3, coords_plane2, ep_x, ep_y, ep_z)

        [phi, u_x, u_y] = calc_min_field(a0, b0, n0, m0, l0, coords_plane2, coords, V_x, V_y, V_x_2, V_y_2)
        arra.append(maxDV(V_x, V_y, u_x, u_y, coords_plane2))

        # Очистка массивов, использовавшихся для расчета полей скоростей
        start()
        
        # Точки В ФСК
        calculate_earth_point(O1, K1, KSC, coords_plane2, e_point_x[it], e_point_y[it], e_point_z[it], (k - it - 1) * t_2)
        earth_dots.append(fsk_dots(fsk_center, FSK,  e_point_x[it], e_point_y[it], e_point_z[it], O1))
        
    if len(prev_a):        
        avg = 0
        for it in range(len(prev_a)):
            avg += (arra[it])
        avg /= len(prev_a)
        research_1[0].append(deg)
        research_1[1].append(avg)
        maxAB.append([max(research_1[1][-1], research_2[1][-1]), deg])
    else:
        avg = 0
       
        for it in range(len(arra)):
            prev_a.append(arra[it])
            avg += (arra[it])
        avg /= len(prev_a)
        research_2[0].append(deg)
        research_2[1].append(avg)
        
    return [fsk_center, fsk_x, FSK, t_2]


def calculate_4(coords_plane2):
    # количество снимков
    k = 5
    
    # вычисление углов
    precision = 1
    for iteration in range(0, 1):
        earth_dots.clear()
        # первоначальные самолетные углы
        prev_a = []
        [pitch_ang, roll_ang, yaw_ang] = [math.radians(iteration / precision), 0, 0]
        # цикл ряда пластин
        for rows in range(1, 3):                
            [p1, p2, p3, p4] = calculate_3(coords_plane2, [pitch_ang, roll_ang, yaw_ang], rows * k, k * (rows - 1), iteration / precision, prev_a)
            # сдвиг пластины для съемки второго ряда
            
            a1 = earth_dots[0][0][2]
            a2 = earth_dots[0][-1][2]
            a3 = math.sqrt((earth_dots[0][0][1] - earth_dots[0][-1][1]) ** 2)
            ang_a = (n0 + 1) * math.acos((-a3 ** 2 + a1 ** 2 + a2 ** 2) / (2 * a1 * a2)) / n0
            
            b1 = earth_dots[0][-1][2]
            b2 = earth_dots[-1][len(earth_dots[-1]) - m0][2]
            b3 = math.sqrt((earth_dots[0][-1][0] - earth_dots[-1][len(earth_dots[-1]) - m0][0]) ** 2)
            ang_b = 0.985 * (m0 + 1) * math.acos((-b3 ** 2 + b1 ** 2 + b2 ** 2) / (2 * b1 * b2)) / m0

            roll_ang += ang_a
            pitch_ang -= ang_b
            
    print(math.degrees(ang_a))
    
    # vectr = []
    # for i in range(len(research_1[1])):
    #     vectr.append((research_1[1][i] + research_2[1][i]) / 2)
    
    # # df_a = pd.read_excel('Results4.xlsx')
    # df = pd.DataFrame({'Угол': research_1[0],
    #                 'Значение': research_1[1],
    #                 'Значение (тангаж 0)': research_2[1], 
    #                 'Max': maxAB})
    # df.to_excel('./Results11.xlsx', index=False)
    
    # best_ang = sorted(maxAB, key = lambda x:x[0])
    # print("Наилучшее значение: ", best_ang[0][0])
    # print("Угол: ", best_ang[0][1])
    # print(earth_dots[0][-1][1] - earth_dots[1][0][1], earth_dots[0][-1][1] - earth_dots[1][0][1])
    earth_dots_picture(earth_dots, p1, p2, p3, p4)
    # plotting(mode)
    
def plotting(mode):
    df_a = pd.read_excel('Results3.xlsx')
    
    maxab = []
    for i in range(len(df_a['Значение'])):
        maxab.append([max(df_a['Значение'][i], df_a['Значение (тангаж 0)'][i]), df_a['Угол'][i]])
        
    best_ang = sorted(maxab, key = lambda x : x[0])
    print("Наилучшее значение: ", best_ang[0][0])
    print("Угол: ", best_ang[0][1])

    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.plot(df_a['Угол'], df_a['Значение'], color = 'black', linewidth = 0.5)
    ax.scatter(df_a['Угол'], df_a['Значение'], color = 'b', s = 20, label = 'Второй ряд')
    
    ax.plot(df_a['Угол'], df_a['Значение (тангаж 0)'], color = 'black', linewidth = 0.5)
    ax.scatter(df_a['Угол'], df_a['Значение (тангаж 0)'], color = 'r', s = 20, label = 'Первый ряд')
    
    ax.set_xlabel('Тангаж [градусы]', fontsize = 20)
    ax.set_ylabel('MaxDV [м]', fontsize = 20)
    
    ite = 0
    while df_a['Угол'][ite] != 5.3:
        ite += 1

    ax.plot(5.3, df_a['Значение (тангаж 0)'][ite], 'ro', color = 'black')
    ax.plot(5.3, df_a['Значение'][ite], 'ro', color = 'black')
    
    arrowprops = {
        'arrowstyle': '->',
    }

    # !!! Добавление аннотации
    plt.annotate('Оптимальные значения', size = 20,
                 xy=(5.3, df_a['Значение (тангаж 0)'][ite]),
                 xytext=(5.3 - 4, df_a['Значение (тангаж 0)'][ite] + 3e-7),
                 arrowprops=arrowprops)
    
    plt.annotate('Оптимальные значения', size = 20,
                 xy=(5.3, df_a['Значение'][ite]),
                 xytext=(5.3 - 4, df_a['Значение (тангаж 0)'][ite] + 3e-7),
                 arrowprops=arrowprops)
    
    fig.suptitle('График зависимости смаза от начального угла тангажа', fontsize=20)
    
    ax.legend(fontsize = 15)
    ax.grid(True)
    plt.title(mode)
    plt.show()

if __name__ == "__main__":
    #выбор режима работы программы
    mode = 'RESEARCH'

    if mode == 'VISUAL':
        # a = 1 #(метры) большая полуось
        # e = 0.001         #Эксцентриситет
        i = 90           #(градусы) наклонение орбиты
        # Ω_0 = 0           #(град) долгота восходящего узла
        # ω_0 = 0   #(град) начальный угол наклона вектора скорости к экватору(аргумент перигея)
        # M_0 = 32.6650111#32.6650111    #(град) средняя аномалия в эпоху кеплеровой орбиты
        t_0 = 26300         #(с) начальный момент времени (в этом месте в книге опечатка(26300)
        t = 45300#2.823   #(c) конечный момент времени u = 0 при 31238.21, u = pi/2 при 27147.68 u = pi при 28522.71, u = 3pi/2 при 29897.75
        step = 100 #Шаг(промежуток)
        α = math.pi / 8
        calculate(mode, None, [0, α, 0])
        dataSet = np.array([longitude[1: len(dolg) - 1], latitude[1: len(shir) - 1]])
        trace_points(step, dataSet_4)
        map_2d(dataSet, x_trace, y_trace)
        animation_3d(step, dataSet_2, dataSet_3, dataSet_4, O2)

    elif mode == 'MIN_FIELD':
        a = 6678000
        e=0.01
        i = 60
        M_0=32.6650111
        t_0=26300
        t=27147.68
        step = 0.01
        a0 = 0.063524#0.072#180#72   #1/2 длины ПЗС линейки по оси Oy(метры)
        b0 = 0.007854#0.0076#40#7.6  #1/2 длины ПЗС линейки по оси Оx(метры)
        n1 = 2 * a0 / (n0 - 1)   #шаг по оси Оy
        m1 = 2 * b0 / (m0 - 1)   #шаг по оси Ox
        k = 3
        # t_0 = 0
        # t = 2.8
        angels = [0, 0, 0]
        
        #для нужд задачи минимизации поля скоростей
        coords_2 = [[0] * m0 for i in range(n0)]
        V_x_2 = [[0] * m0 for i in range(n0)]
        V_y_2 = [[0] * m0 for i in range(n0)]
        scale_prm = 0.01
        scale_prm = scale_prm * 10 ** k
        a0, b0, n1, m1, a, f, d, R = conversion(k, a, f, d, R, a0, b0, n1, m1)
        
        for it in np.linspace(-a0, a0, n0):
            for j in np.linspace(-b0, b0, m0):
                coords_plane2.append([round(j, 10), round(it, 10)])
                
        calculate(mode, coords_plane2, angels)
        [phi, u_x, u_y] = calc_min_field(a0, b0, n0, m0, l0, coords_plane2, coords_2, V_x, V_y, V_x_2, V_y_2)
        print(phi, u_x, u_y)
        speed_vector(a0, b0, n0, m0, n1, m1, l0, V_x, V_y, u_x, u_y, phi, coords_plane2, scale_prm)
        print(a0 / (R * (n_0 - w_z)))

    elif mode == 'HALF_PLATE':
        a0 = 0.063524#0.072#180#72   #1/2 длины ПЗС линейки по оси Oy(метры)
        b0 = 0.007854#0.0076#40#7.6  #1/2 длины ПЗС линейки по оси Оx(метры)
        n1 = 2 * a0 / (n0 - 1)   #шаг по оси Оy
        m1 = 2 * b0 / (m0 - 1)   #шаг по оси Ox
        # i = 0.00000000001 #(град) наклонение орбиты
        # t = 0.002
        k = 3
        a0, b0, n1, m1, a, f, d, R = conversion(k, a, f, d, R, a0, b0, n1, m1)
        p3 = Plates(a0, b0, n0, m0, n1, m1, 5)
        calculate_2(mode, p3.coords)
    elif mode == 'ROLL+PITCH_AXES':
        a0 = 0.0125#0.072#180#72   #1/2 длины ПЗС линейки по оси Oy(метры)
        b0 = 0.007854#0.0076#40#7.6  #1/2 длины ПЗС линейки по оси Оx(метры)
        n1 = 2 * a0 / (n0 - 1)   #шаг по оси Оy
        m1 = 2 * b0 / (m0 - 1)   #шаг по оси Ox
        i = 0.00000000001 #(град) наклонение орбиты
        t = 0.002
        k = 3
        a0, b0, n1, m1, a, f, d, R = conversion(k, a, f, d, R, a0, b0, n1, m1)
        p3 = Plates(a0, b0, n0, m0, n1, m1, 5)
        calculate_2(mode, p3.coords)
    elif mode == 'RESEARCH':
        a = 6678245 #(метры) большая полуось
        # e = 0#0.00000000001         #Эксцентриситет
        # i = 0.00000000001           #(градусы) наклонение орбиты
        # Ω_0 = 0           #(град) долгота восходящего узла
        # ω_0 = 0   #(град) начальный угол наклона вектора скорости к экватору(аргумент перигея)
        # M_0 = 45#32.6650111    #(град) средняя аномалия в эпоху кеплеровой орбиты
        t_0 = 0#26300         #(с) начальный момент времени (в этом месте в книге опечатка(26300)
        t = 2.823#28500#2.823  
        i = 0.001
        a0 = 0.018#0.072#180#72   #1/2 длины ПЗС линейки по оси Oy(метры)
        n0 = 13
        m0 = 5
        t = 0.002
        # b0 = 0.007854#0.0076#40#7.6  #1/2 длины ПЗС линейки по оси Оx(метры)
        b0 = 0.0039
        n1 = 2 * a0 / (n0 - 1)   #шаг по оси Оy
        m1 = 2 * b0 / (m0 - 1)   #шаг по оси Ox
        # i = 0.00000000001 #(град) наклонение орбиты
        # k = 3
                
        for it in np.linspace(-a0, a0, n0):
            for j in np.linspace(-b0, b0, m0):
                coords_plane2.append([round(j, 10), round(it, 10)])
        calculate_4(coords_plane2)
    else:
        None