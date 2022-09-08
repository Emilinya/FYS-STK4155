from cgi import test
import numpy as np

def power_itr(d):
    for c in range(d+1):
        for y in range(c+1):
            for x in range(c+1):
                if x + y == c:
                    yield (x, y)

def poly_generator(d, beta):
    if len(list(poly_generator(d))) != len(beta):
        print(">:(")
        exit(1)

    def p(x, y):
        return np.sum([beta[i] * x**xd * y**yd for i, (xd, yd) in enumerate(power_itr(d))])
    p = np.vectorize(p)
    return p

def test_power_itr(d):
    power_strs = []
    for (x, y) in power_itr(d):
        if x == 0 and y == 0:
            power_strs.append("1")
            continue
        xsym = f"x^{x}" if x > 1 else "x" if x == 1 else ""
        ysym = f"y^{y}" if y > 1 else "y" if y == 1 else ""
        power_strs.append(xsym+ysym)

    print(f"P_{d}(x, y) = " + " + ".join(power_strs))

if __name__ == "__main__":
    test_power_itr(4)