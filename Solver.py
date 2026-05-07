from Givence import Givence_rotation
from Matrix import Matrix
import copy

EPS = 1e-20

def Solver_QR_Decomposition(A, f):
    Q, R = Givence_rotation(A)

    # Прямой ход y = Q^T * f
    Q.transpose()
    y = Q.multiply_matrix(f)

    #Обратный ход Rx=y
    x = Matrix(data=[[0]*R.rows]) #вектор решений
    for i in range(R.rows-1, -1, -1):
        summ = 0 #сумма элементов которые мы уже нашли
        for j in range(i+1, R.cols):
            summ += R.data[j][i]*x.data[0][j]

        x.data[0][i] = (y.data[0][i]-summ)/R.data[i][i]

    return x

#Поиск ведущего элемента
def Find_Main_Element(A, j):
    M = A.rows
    index = j
    for i in range(j+1, M):
        if (abs(A.data[j][i]) > abs(A.data[j][index])):
            index = i

    return index

def Direct_way(A, f):
    M = A.rows

    for i in range(M - 1):
        I = Find_Main_Element(A, i)

        # Перестановка строк
        if I != i:
            for k in range(A.cols):
                A.data[k][i], A.data[k][I] = A.data[k][I], A.data[k][i]

            f.data[0][i], f.data[0][I] = f.data[0][I], f.data[0][i]

        # Проверка на ноль
        if abs(A.data[i][i]) < EPS:
            raise ValueError("Нулевой ведущий элемент")

        # Прямой ход
        for j in range(i + 1, M):
            factor = A.data[i][j] / A.data[i][i]

            for k in range(i, A.cols):
                A.data[k][j] -= factor * A.data[k][i]

            f.data[0][j] -= factor * f.data[0][i]

    return A, f

def Back_Row_Substitution(A, f):
    M = A.rows
    if M == 0:
        raise ValueError('Back_Row_Substitution: empty matrix')
    if M != A.cols:
        raise ValueError('Back_Row_Substitution: not square matrix')

    res = copy.deepcopy(f)

    for i in range(M-1, -1, -1):
        if abs(A.data[i][i]) < EPS:
            raise ValueError('Back_Row_Substitution: division by zero')

        for j in range(i+1, M):
            res.data[0][i] -= A.data[j][i] * res.data[0][j]
        res.data[0][i] /= A.data[i][i]

    return res


def Solver_Gauss(A, f):
    M = A.rows
    if M == 0:
        raise ValueError('Solver_Gauss: the matrix A is empty')
    if M != A.cols:
        raise ValueError('Solver_Gauss: the matrix A is not square')

    A2, f2 = Direct_way(A, f)

    res = Back_Row_Substitution(A2, f2)

    return res

