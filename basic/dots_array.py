from src import *
from basic.earth_inter import earth_inter

def dots_array(x_rv, y_rv, z_rv, r, O1, OSC, K1, KSC):
    #Массив точки спутника в прямоугольных гринвических координатах
    dataSet_2[0].append(np.array([x_rv]))
    dataSet_2[1].append(np.array([y_rv]))
    dataSet_2[2].append(np.array([z_rv]))
    
    #Массивы проекции спутника на поверхность Земли в прямоугольных гринвических координатах
    dataSet_3[0].append(np.array([x_rv * R / r]))
    dataSet_3[1].append(np.array([y_rv * R / r]))
    dataSet_3[2].append(np.array([z_rv * R / r]))
    
    dir = (O1 - K1)
    dir /= np.linalg.norm(dir)
    tmp = min(earth_inter(K1, dir)[0], earth_inter(K1, dir)[1])
    if(tmp > 0):
        #Проекция спутника на земле
        POINT = K1 + dir * tmp

        dataSet_4[0].append(np.array(POINT[0]))
        dataSet_4[1].append(np.array(POINT[1]))
        dataSet_4[2].append(np.array(POINT[2]))
    else:
        None