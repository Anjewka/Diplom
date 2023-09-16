import math
from src import *
import numpy as np
from basic.calc_KSK import calc_KSK

def calc_grinvich_coords(t_1, a, e, i, t_0, step, d):
    #(градусы) средняя аномалия на заданный момент времени
    #функцией math.degrees(x) переводим значение x из радиан в градусы
    M = M_0 + math.degrees(n_0 * (t_1 - t_0))
    #print("Средняя аномалия в момент времени t_1 (град):", M)

    #(град)эксцентрическая аномалия
    #задавшись точностью вычислений ε(град), методом последовательных приближений вычисляем
    #эксцентрическую аномалию
    ε = 1e-9
    E_0 = M + math.degrees(e * math.sin(math.radians(M)))
    E = M + math.degrees(e * math.sin(math.radians(E_0)))
    while abs(E_0 - E) > ε:
        E_0 = E
        E = M + math.degrees(e * math.sin(math.radians(E)))

    #(град) истинная аномалия
    ξ = E + 2 * math.degrees(math.atan(e * math.sin(math.radians(E)) / (1 + math.sqrt(1 - e ** 2) - e * math.cos(math.radians(E)))))
    #print("Истинная аномалия(град)", ξ)

    #(град) аргумент широты
    u = ω_0 + ξ
    #print("Аргумент широты(град):", u)

    #(милиметры) геоцентрическое расстояние до спутника
    r = p / (1 + e * math.cos(math.radians(ξ)))
    
    #получим прямоугольные гринвические координаты(милиметры)
    x_rv = r * (math.cos(math.radians(Ω_0)) * math.cos(math.radians(u)) - math.sin(math.radians(Ω_0)) * math.sin(math.radians(u)) * math.cos(math.radians(i)))
    y_rv = r * (math.sin(math.radians(Ω_0)) * math.cos(math.radians(u)) + math.cos(math.radians(Ω_0)) * math.sin(math.radians(u)) * math.cos(math.radians(i)))
    z_rv = r * math.sin(math.radians(u)) * math.sin(math.radians(i))
    
    #(градусы) геоцентрическая широта (спутника и подспутниковой точки)
    sh = math.degrees(math.atan(z_rv / math.sqrt(x_rv ** 2 + y_rv ** 2)))
    latitude.append(sh)
    #print(sh)

    #(град) дуга AB
    if (u % 360 >= 0 and u % 360 <= 90):
        AB = math.degrees(math.asin(math.tan(math.radians(sh)) / math.tan(math.radians(i))))
    elif (u % 360 > 90 and u % 360 <= 270):
        AB = 180 - math.degrees(math.asin(math.tan(math.radians(sh)) / math.tan(math.radians(i))))
    else: 
        AB = math.degrees(math.asin(math.tan(math.radians(sh)) / math.tan(math.radians(i))))

    #(град) долгота подспутниковой точки(в гринвической СК)
    do = Ω_0 + AB
    do_2d = do - math.degrees(w_z * (t_1 - t_0 + step))
    do_2d = do_2d % 360
    if do_2d >= 180 and do_2d <= 360:
        do_2d = do_2d - 360
    longitude.append(do_2d)

    #разбираемся с системами координат
    O1 = np.array([x_rv, y_rv, z_rv])                            #центр ОСК(ССК) в ИСК
    e3 = np.copy(O1)
    e3 = e3 / np.linalg.norm(e3)

    e2 = np.array([-math.sin(math.radians(Ω_0)) * math.sin(math.radians(i)), 
    math.cos(math.radians(Ω_0)) * math.sin(math.radians(i)), 
    -math.cos(math.radians(i))])

    e3 = -e3
    e1 = np.cross(e2, e3) 
    e1 = e1 / np.linalg.norm(e1)
    OSC = np.array([e1, e2, e3]) #Орбитальная система координат
    OSC = OSC / np.linalg.norm(OSC)
    O2.append(O1)

    # нормирование столбцов
    norms = np.linalg.norm(OSC, axis=0)
    OSC = OSC / norms
    
    K1, KSC = calc_KSK(0, 0, 0, O1, OSC, r, d) 
    
    OSC_Arr.append(KSC)

    return [x_rv, y_rv, z_rv, r, OSC, O1, e3]