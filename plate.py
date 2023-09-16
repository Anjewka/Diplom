import numpy as np
from speed_field.conversion import conversion

class Plate(object):

    def __init__(self, a0, b0, n0, m0, n1, m1, dy, dx):  #a0 - 1/2 длины пластины по OY, b0 - по OX, n0 и m0 - количество точек по OY и OX соответственно
        self.coords = []
        for it in np.linspace(-a0, a0, n0):      #dx, dy смещение центра по OX и OY соответственно
            for j in np.linspace(-b0, b0, m0):
                self.coords.append([round(j, 10) + dx, round(it, 10) + dy])


class Plates(object):
    coords = []

    def __init__(self, a0, b0, n0, m0, n1, m1, q):      #q - количество микросхем
        if q == 5:
            p1 = Plate(a0, b0, n0, m0, n1, m1, -4 * a0 - 2 * n1, b0 + 0.5 * m1)
            p2 = Plate(a0, b0, n0, m0, n1, m1, -2 * a0 - n1, -b0 - 0.5 * m1)
            p3 = Plate(a0, b0, n0, m0, n1, m1, 0, b0 + 0.5 * m1)
            p4 = Plate(a0, b0, n0, m0, n1, m1, 2 * a0 + n1, -b0 - 0.5 * m1)
            p5 = Plate(a0, b0, n0, m0, n1, m1, 4 * a0 + 2 * n1, b0 + 0.5 * m1)
            self.coords = p1.coords + p2.coords + p3.coords + p4.coords + p5.coords