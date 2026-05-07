from Matrix import Matrix
from Hausholder import Hausholder_Row_Transformation, Hausholder_Column_Transformation

EPS = 1e-20

def SVD(inpit_matrix, U, Sigma, V):
    M = inpit_matrix.rows
    if M == 0:
        raise ValueError('SVD: the matrix is empty')
    N = inpit_matrix.cols
    if N == 0:
        raise ValueError('SVD: the matrix is empty')

    Min_size = min(M, N)
    Up_size = Min_size-1
    Down_size = Min_size-1

    U.resize(M, M)
    Sigma.resize(M, N)
    V.resize(N, N)

    for i in range(M):
        U.data[i][i] = 1.0
        for j in range(N):
            Sigma.data[j][i] = inpit_matrix.data[j][i]
    for i in range(N):
        V.data[i][i] = 1.0

    #Этап 1: бидиагонализация
    for i in range(Min_size-1):
        Hausholder_Column_Transformation(Sigma, U, i, i)
        Hausholder_Row_Transformation(Sigma, V, i, i+1)

    if M > N:
        Hausholder_Column_Transformation(Sigma, U, N-1, N-1)
        Down_size += 1
    elif N > M:
        Hausholder_Row_Transformation(Sigma, V, M-1, M)
        Up_size += 1

    #Этап 2: приведение матрицы к диагональному виду
    while True:
        CountUpElements = 0
        for i in range(Up_size):
            if abs(Sigma.data[i+1][i]) > EPS:
                Hausholder_Row_Transformation(Sigma, V, i, i)
            else:
                CountUpElements += 1

        if CountUpElements == Up_size:
            break

        for i in range(Down_size):
            if abs(Sigma.data[i][i+1]) > EPS:
                Hausholder_Column_Transformation(Sigma, U, i, i)

    for i in range(Min_size):
        if Sigma.data[i][i] < 0:
            Sigma.data[i][i] = -Sigma.data[i][i]
            for j in range(M):
                U.data[i][j] = -U.data[i][j]

    for I in range(Min_size):
        Max_Elem = Sigma.data[I][I]
        Index = I
        for i in range(I+1, Min_size):
            if (Sigma.data[i][i] > Max_Elem):
                Max_Elem = Sigma.data[i][i]
                Index = i
        if I != Index:
            Sigma.data[Index][Index] = Sigma.data[I][I]
            Sigma.data[I][I] = Max_Elem
            for row in range(M):
                elem = U.data[I][row]
                U.data[I][row] = U.data[Index][row]
                U.data[row][Index] = elem
            for i in range(N):
                elem = V.data[I][row]
                V.data[I][row] = V.data[Index][row]
                V.data[Index][row] = elem

    return U, Sigma, V

def Solver_SVD(A, f):
    U, Sigma, V = Matrix(data=[[0]*A.rows for i in range(A.cols)]),  Matrix(data=[[0]*A.rows for i in range(A.cols)]),  Matrix(data=[[0]*A.rows for i in range(A.cols)])
    U, Sigma, V = SVD(A, U, Sigma, V)
    Min_size = min(Sigma.rows, Sigma.cols)
    print(f"Cond(A)={Sigma.data[0][0]/Sigma.data[Min_size-1][Min_size-1]}")

    for i in range(Min_size): #проверка на усечение
        if Sigma.data[i][i] < 1e-10:
            Min_size = i
            break

    Sigma.resize(Min_size, Min_size)
    U.resize(U.cols, Min_size)
    V.resize(V.cols, Min_size)

    U.transpose()
    Ut_f = U.multiply_matrix(f)
    for i in range(Min_size):
        Ut_f.data[0][i] /= Sigma.data[i][i]
    result = V.multiply_matrix(Ut_f)

    return result

