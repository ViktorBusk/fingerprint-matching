import  numpy as np

def mean(I):
    return 1 / (I.shape[0]*I.shape[1]) * np.sum(I)

def variance(I):
    MI = mean(I)
    I_new = np.empty(I.shape)
    I_new = (I - MI)**2
    return 1 / (I.shape[0]*I.shape[1]) * np.sum(I_new)

def normalize(I, M0: int, VAR0: int):
    G = np.empty(I.shape)
    M = mean(I)
    VAR = variance(I)

    for i in range(I.shape[1]):
        for j in range(I.shape[0]):
            if I[j, i] > M:
                G[j, i] = (M0 + np.sqrt(VAR0 * (I[j, i] - M)**2 / VAR))
            else:
                G[j, i] = (M0 - np.sqrt(VAR0 * (I[j, i] - M)**2 / VAR))
    return G
