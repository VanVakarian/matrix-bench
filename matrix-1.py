import random
import time
import math

# BATCH_SIZE = 1000
# INPUT_SIZE = 784
# HIDDEN_SIZE = 512
# OUTPUT_SIZE = 10
# ITERATIONS = 50

BATCH_SIZE = 1000
INPUT_SIZE = 78
HIDDEN_SIZE = 51
OUTPUT_SIZE = 10
ITERATIONS = 50

def create_matrix(rows, cols, fill=None):
    if fill is None:
        return [[random.gauss(0, 1) for _ in range(cols)] for _ in range(rows)]
    return [[fill for _ in range(cols)] for _ in range(rows)]

def create_vector(size, fill=None):
    if fill is None:
        return [random.gauss(0, 1) for _ in range(size)]
    return [fill for _ in range(size)]

def matrix_multiply(a, b):
    rows_a = len(a)
    cols_a = len(a[0])
    cols_b = len(b[0])
    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a):
                result[i][j] += a[i][k] * b[k][j]
    return result

def transpose(matrix):
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def add_matrix(a, b):
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

def subtract_matrix(a, b):
    return [[a[i][j] - b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

def add_vector_to_matrix(matrix, vector):
    return [[matrix[i][j] + vector[j] for j in range(len(matrix[0]))] for i in range(len(matrix))]

def relu(matrix):
    return [[max(0, val) for val in row] for row in matrix]

def relu_derivative(matrix):
    return [[1 if val > 0 else 0 for val in row] for row in matrix]

def softmax(matrix):
    result = [[0 for _ in range(len(matrix[0]))] for _ in range(len(matrix))]
    for i in range(len(matrix)):
        row_max = max(matrix[i])
        exp_sum = sum(math.exp(x - row_max) for x in matrix[i])
        for j in range(len(matrix[0])):
            result[i][j] = math.exp(matrix[i][j] - row_max) / exp_sum
    return result

def hadamard_product(a, b):
    return [[a[i][j] * b[i][j] for j in range(len(a[0]))] for i in range(len(a))]

def sum_columns(matrix):
    return [sum(row[j] for row in matrix) for j in range(len(matrix[0]))]

def generate_data():
    X = create_matrix(BATCH_SIZE, INPUT_SIZE)
    y = [[0 for _ in range(OUTPUT_SIZE)] for _ in range(BATCH_SIZE)]
    for i in range(BATCH_SIZE):
        target = random.randint(0, OUTPUT_SIZE - 1)
        y[i][target] = 1
    return X, y

def forward(X, W1, b1, W2, b2):
    z1 = add_vector_to_matrix(matrix_multiply(X, W1), b1)
    a1 = relu(z1)
    z2 = add_vector_to_matrix(matrix_multiply(a1, W2), b2)
    probs = softmax(z2)
    return z1, a1, z2, probs

def backward(X, y, probs, z1, a1, W1, W2):
    dz2 = subtract_matrix(probs, y)
    dW2 = matrix_multiply(transpose(a1), dz2)
    db2 = sum_columns(dz2)

    da1 = matrix_multiply(dz2, transpose(W2))
    dz1 = hadamard_product(da1, relu_derivative(z1))
    dW1 = matrix_multiply(transpose(X), dz1)
    db1 = sum_columns(dz1)

    return dW1, db1, dW2, db2

def init_weights():
    scale1 = math.sqrt(2.0/INPUT_SIZE)
    scale2 = math.sqrt(2.0/HIDDEN_SIZE)
    W1 = [[random.gauss(0, 1) * scale1 for _ in range(HIDDEN_SIZE)] for _ in range(INPUT_SIZE)]
    b1 = create_vector(HIDDEN_SIZE, 0)
    W2 = [[random.gauss(0, 1) * scale2 for _ in range(OUTPUT_SIZE)] for _ in range(HIDDEN_SIZE)]
    b2 = create_vector(OUTPUT_SIZE, 0)
    return W1, b1, W2, b2

def main():
    W1, b1, W2, b2 = init_weights()
    X, y = generate_data()

    forward_time = 0
    backward_time = 0

    for _ in range(ITERATIONS):
        start = time.perf_counter()
        z1, a1, z2, probs = forward(X, W1, b1, W2, b2)
        forward_time += time.perf_counter() - start

        start = time.perf_counter()
        dW1, db1, dW2, db2 = backward(X, y, probs, z1, a1, W1, W2)
        backward_time += time.perf_counter() - start

    print(f'Размеры матриц:')
    print(f'Батч: {BATCH_SIZE}')
    print(f'Вход: {INPUT_SIZE}')
    print(f'Скрытый слой: {HIDDEN_SIZE}')
    print(f'Выход: {OUTPUT_SIZE}')
    print(f'\nВремя выполнения ({ITERATIONS} итераций):')
    print(f'Forward pass: {forward_time/ITERATIONS:.6f} сек')
    print(f'Backward pass: {backward_time/ITERATIONS:.6f} сек')
    print(f'Общее время: {(forward_time + backward_time)/ITERATIONS:.6f} сек')

if __name__ == '__main__':
    main()
