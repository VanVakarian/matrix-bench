import torch
import torch.nn as nn
import time

# BATCH_SIZE = 1000
# INPUT_SIZE = 784
# HIDDEN_SIZE = 512
# OUTPUT_SIZE = 10
# ITERATIONS = 50

BATCH_SIZE = 2150
INPUT_SIZE = 1686
HIDDEN_SIZE = 1100
OUTPUT_SIZE = 10
ITERATIONS = 50


class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        self.softmax = nn.Softmax(dim=1)

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        z1 = self.fc1(x)
        a1 = self.relu(z1)
        z2 = self.fc2(a1)
        probs = self.softmax(z2)
        return z1, a1, z2, probs


def generate_data():
    X = torch.randn(BATCH_SIZE, INPUT_SIZE)
    y = torch.randint(0, OUTPUT_SIZE, (BATCH_SIZE,))
    y_one_hot = torch.zeros(BATCH_SIZE, OUTPUT_SIZE)
    y_one_hot.scatter_(1, y.unsqueeze(1), 1)
    return X, y_one_hot


def main():
    model = SimpleNN()
    criterion = nn.MSELoss()
    X, y = generate_data()

    forward_time = 0
    backward_time = 0

    for _ in range(ITERATIONS):
        start = time.perf_counter()
        z1, a1, z2, probs = model(X)
        loss = criterion(probs, y)
        forward_time += time.perf_counter() - start

        start = time.perf_counter()
        loss.backward()
        backward_time += time.perf_counter() - start

        model.zero_grad()

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
