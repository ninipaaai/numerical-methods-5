from Matrix import Matrix
from Givence import Givence_rotation
from Solver import Solver_QR_Decomposition, Solver_Gauss
from Hausholder import Hausholder_Column_Transformation, Hausholder_Row_Transformation
from SVD_Solver import SVD, Solver_SVD
import copy

def norma(x):
    return sum([x.data[0][i]**2 for i in range(x.rows)])**0.5

print('Практическое задание №5   Обухов Назар ПМИ-41')
print()

#Цикл, который решает СЛАУ для каждого порядка матрицы А
a = 1.218 #15 вариант
b = 3.493
for k in [5, 10, 20]:
    x = Matrix(data=[[1] * k])
    A = Matrix(data=[[0]*k for i in range(k)])
    for j in range(k):
        for i in range(k):
            A.data[i][j] = 1/(1+a*(i+1)+b*(j+1)) #формируем матрицу А
    f = A.multiply_matrix(x) #вектор свободных членов

    print('Порядок', k)
    Ag = copy.deepcopy(A)
    fg = copy.deepcopy(f)
    result1 = Solver_Gauss(Ag, fg)
    for j in range(result1.rows):
        result1.data[0][j] -= 1
    print('Метод Гаусса: ', norma(result1) / norma(x))

    Aq = copy.deepcopy(A)
    fq = copy.deepcopy(f)
    result2 = Solver_QR_Decomposition(Aq, fq)
    for j in range(result2.rows):
        result2.data[0][j] -= 1
    print('QR-разложение: ', norma(result2) / norma(x))

    As = copy.deepcopy(A)
    fs = copy.deepcopy(f)
    result3 = Solver_SVD(As, fs)
    for j in range(result3.rows):
        result3.data[0][j] -= 1
    print('SVD: ', norma(result3) / norma(x))
    print()




