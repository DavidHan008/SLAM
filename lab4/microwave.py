import math

def rou(Pmax, Pmin):

    return 10 **((Pmax - Pmin) / 20)

def B(rou):
    return (rou - 1)/math.sqrt(rou)

def tou(rou):
    return (rou - 1)/(rou + 1)

if __name__ == "__main__":
    Pmax = -39.74
    Pmin = -50.00
    rou1 = 2.89
    rou2 = 3.26

    result = rou(Pmax, Pmin)
    tou1 = tou(rou1)
    tou2 = tou(rou2)
    
    

    print(result)