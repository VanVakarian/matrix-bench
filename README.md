# Тестирование производительности матричных операций

Проект для сравнения производительности матричных операций в различных реализациях: нативный JavaScript, TensorFlow.js, нативный Python, NumPy и PyTorch.

## Установка и настройка окружения

### JavaScript (Node.js)

Инициализация проекта:

```bash
npm init -y
```

Установка зависимостей:

```bash
npm install @tensorflow/tfjs-node
```

### Python

Создание виртуального окружения:

```bash
python -m venv venv
```

Активация виртуального окружения:

Windows:

```bash
.\venv\Scripts\activate
```

Unix/macOS:

```bash
source venv/bin/activate
```

Установка зависимостей:

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Структура проекта

```
.
├── README.md
├── package.json
├── matrix-1.js           # Нативная JS реализация
├── matrix-1-tf.js        # TensorFlow.js реализация
├── matrix-1.py           # Нативная Python реализация
├── matrix-1-np.py        # NumPy реализация
└── matrix-1-pt.py        # PyTorch реализация
```

## Запуск тестов

JavaScript:

```bash
node matrix-1.js          # Нативный JS
node matrix-1-tf.js       # TensorFlow.js
```

Python:

```bash
python matrix-1.py        # Нативный Python
python matrix-1-np.py     # NumPy
python matrix-1-pt.py     # PyTorch
```

## Параметры тестирования

- Размер батча: 1000 (нативные реализации) / 2150 (фреймворки)
- Размер входного слоя: 78/1686
- Размер скрытого слоя: 51/1100
- Размер выходного слоя: 10
- Количество итераций: 50

Параметры можно изменить в начале каждого файла в блоке конфигурации.
