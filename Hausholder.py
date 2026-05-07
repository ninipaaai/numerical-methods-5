from Matrix import Matrix
import math

EPS = 1e-20

def Hausholder_Column_Transformation(A, U, i, j):
    M = A.rows
    N = A.cols

    p = [0]*M
    s, beta, mu = 0.0, 0.0, 0.0

    s = 0
    for I in range(j+1, M):
        s += A.data[i][I]**2

    if math.sqrt(s) > EPS:
        s += A.data[i][j]**2
        if A.data[i][j] < 0:
            beta = math.sqrt(s)
        else:
            beta = -math.sqrt(s)

        mu = 1.0/beta/(beta-A.data[i][j])

        for I in range(M):
            p[I] = 0
            if I >= j:
                p[I] = A.data[i][I]

        p[j] -= beta

        for m in range(N):
            s = 0
            for n in range(j, M):
                s += A.data[m][n]*p[n]
            s *= mu
            for n in range(j, M):
                A.data[m][n] -= s*p[n]

        for m in range(M):
            s = 0
            for n in range(j, M):
                s += U.data[n][m]*p[n]
            s *= mu
            for n in range(j, M):
                U.data[n][m] -= s*p[n]

    return U, A

def Hausholder_Row_Transformation(A, V, i, j):
    M = A.rows
    N = A.cols

    p = [0]*N

    s, beta, mu = 0.0, 0.0, 0.0

    s = 0
    for I in range(j+1, N):
        s += A.data[I][i]**2

    if math.sqrt(s) > EPS:
        s += A.data[j][i]**2
        if A.data[j][i] < 0:
            beta = math.sqrt(s)
        else:
            beta = -math.sqrt(s)

        mu = 1.0/beta/(beta-A.data[j][i])

        for I in range(N):
            p[I] = 0
            if I >= j:
                p[I] = A.data[I][i]
        p[j] -= beta

        for m in range(N):
            s = 0
            for n in range(j, N):
                s += A.data[n][m]*p[n]
            s *= mu
            for n in range(j, N):
                A.data[n][m] -= s*p[n]

        for m in range(N):
            s = 0
            for n in range(j, N):
                s += V.data[n][m]*p[n]
            s *= mu
            for n in range(j, N):
                V.data[n][m] -= s*p[n]

    return V, A




