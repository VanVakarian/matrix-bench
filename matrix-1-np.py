import numpy as np
import time

BATCH_SIZE = 1000
INPUT_SIZE = 784
HIDDEN_SIZE = 512
OUTPUT_SIZE = 10
ITERATIONS = 50

# BATCH_SIZE = 1000
# INPUT_SIZE = 78
# HIDDEN_SIZE = 51
# OUTPUT_SIZE = 10
# ITERATIONS = 50


def generate_data():
    X = np.random.randn(BATCH_SIZE, INPUT_SIZE)
    y = np.random.randint(0, OUTPUT_SIZE, size=(BATCH_SIZE,))
    y_one_hot = np.zeros((BATCH_SIZE, OUTPUT_SIZE))
    y_one_hot[np.arange(BATCH_SIZE), y] = 1
    return X, y_one_hot


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def forward(X, W1, b1, W2, b2):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return z1, a1, z2, probs


def backward(X, y, probs, z1, a1, W1, W2):
    dz2 = probs - y
    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0)

    da1 = dz2 @ W2.T
    dz1 = da1 * relu_derivative(z1)
    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0)

    return dW1, db1, dW2, db2


def main():
    W1 = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * np.sqrt(2.0/INPUT_SIZE)
    b1 = np.zeros(HIDDEN_SIZE)
    W2 = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * np.sqrt(2.0/HIDDEN_SIZE)
    b2 = np.zeros(OUTPUT_SIZE)

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
