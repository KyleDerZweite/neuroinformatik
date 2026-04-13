import random
import math

VECTOR = list[float]
MATRIX = list[VECTOR]

def create_weight_matrix(rows: int, cols: int) -> MATRIX:
    """
    Introduced Xavier initialization to maintain stable variance of activations and gradients across layers.

    Glorot, X., & Bengio, Y. (2010). Understanding the difficulty of training deep feedforward neural networks 
    - https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf
    
    Tbh, kein plan, passt und hinterfrag ich nicht weiter. 
    Ich weiß nicht wie es hilft, aber es soll helfen das die werte nicht zu klein oder groß werden. 
    """
    limit = math.sqrt(6 / (rows + cols))
    return [[random.uniform(-limit, limit) for _ in range(cols)] for _ in range(rows)]

def create_bias_vector(size: int) -> VECTOR:
    return [0.0 for _ in range(size)]

def element_wise_add(a: MATRIX, b: MATRIX) -> MATRIX:
    for i in range(len(a)):
        for j in range(len(a[0])):
            a[i][j] += b[i][j]
    return a

def element_wise_subtract(a: MATRIX, b: MATRIX) -> MATRIX:
    for i in range(len(a)):
        for j in range(len(a[0])):
            a[i][j] -= b[i][j]
    return a

def element_wise_multiply(a: MATRIX, b: MATRIX) -> MATRIX:
    for i in range(len(a)):
        for j in range(len(a[0])):
            a[i][j] *= b[i][j]
    return a

def dot(a: MATRIX, b: MATRIX) -> MATRIX:
    result = [[0.0 for _ in range(len(b[0]))] for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                result[i][j] += a[i][k] * b[k][j]
    return result

def print_matrix(matrix: MATRIX) -> None:
    for row in matrix:
        print(" ".join(f"{value}" for value in row))
