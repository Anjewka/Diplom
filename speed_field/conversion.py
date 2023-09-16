#перевод метры в метры/10**k
def conversion(k, a, f, d, R, a0, b0, n1, m1):
    a = a * 10 ** k
    f = f * 10 ** (k**3)
    d = d * 10 ** (k)
    a0 = a0 * 10 ** k
    b0 = b0 * 10 ** k
    n1 = n1 * 10 ** k
    m1 = m1 * 10 ** k
    R = R * 10 ** k

    return a0, b0, n1, m1, a, f, d, R