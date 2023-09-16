import math
from src import x_trace, y_trace, R, w_z

#Координыт точек трассы спутника фотоаппарата
def trace_points(step, dataSet_4):
    for it in range(1, len(dataSet_4[1])):
        if(dataSet_4[2][it] and dataSet_4[1][it]):
            y_a = math.asin(dataSet_4[2][it] / R)
            x_a = math.asin(dataSet_4[1][it] / (R * math.cos(y_a)))

            if it == 0:
                x_a = math.atan2(dataSet_4[1][it], dataSet_4[0][it])

            y_a = math.degrees(y_a)
            x_a = math.degrees(x_a)

            if dataSet_4[1][it] > 0 and dataSet_4[0][it] < 0:
                x_a = -x_a - 180

            if dataSet_4[1][it] < 0 and dataSet_4[0][it] < 0:
                x_a = -x_a + 180

            if x_a > 180:
                x_a = x_a - 360
            if x_a < - 180:
                x_a = x_a + 360

            x_trace.append(x_a)
            y_trace.append(y_a)

    for it in range(len(x_trace)):
        x_trace[it] = x_trace[it] - math.degrees(w_z * step * (it + 1))
        if x_trace[it] < - 180:
            x_trace[it] = x_trace[it] + 360