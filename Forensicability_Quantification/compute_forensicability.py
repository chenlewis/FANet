import math

e_c = [[0.5883, 0.9642], [0.6585, 0.0800], [0.1873, 0.5226]]
Lambda_c = [0.3852, 0.4049, 0.6413]
beta = 0.5
sigma = 0.1

def L1_distance(a, b):
    return abs(a - b)

def L2_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def K_e3(y, e_c, Lambda, sigma):
    return math.exp(-(Lambda[2] * L2_distance(y, e_c[2])) ** 2 / (2 * sigma ** 2))

def F(y, beta):
    return beta * (1 - K_e3(y, e_c, Lambda_c, sigma)) + (1 - beta) * L1_distance(L2_distance(y, e_c[0]), L2_distance(y, e_c[1]))

if __name__ == "__main__":
    print(F([1.0, 1.0], 0.5))