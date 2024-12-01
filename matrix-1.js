// const BATCH_SIZE = 1000;
// const INPUT_SIZE = 784;
// const HIDDEN_SIZE = 512;
// const OUTPUT_SIZE = 10;
// const ITERATIONS = 50;

const BATCH_SIZE = 1000;
const INPUT_SIZE = 78;
const HIDDEN_SIZE = 51;
const OUTPUT_SIZE = 10;
const ITERATIONS = 50;

function createMatrix(rows, cols, fill = null) {
  if (fill === null) {
    return Array(rows)
      .fill()
      .map(() =>
        Array(cols)
          .fill()
          .map(() => randn())
      );
  }
  return Array(rows)
    .fill()
    .map(() => Array(cols).fill(fill));
}

function createVector(size, fill = null) {
  if (fill === null) {
    return Array(size)
      .fill()
      .map(() => randn());
  }
  return Array(size).fill(fill);
}

function randn() {
  let u = 0,
    v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

function matrixMultiply(a, b) {
  const rows = a.length;
  const cols = b[0].length;
  const result = Array(rows)
    .fill()
    .map(() => Array(cols).fill(0));

  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) {
      for (let k = 0; k < b.length; k++) {
        result[i][j] += a[i][k] * b[k][j];
      }
    }
  }
  return result;
}

function transpose(matrix) {
  return matrix[0].map((_, i) => matrix.map((row) => row[i]));
}

function addMatrix(a, b) {
  return a.map((row, i) => row.map((val, j) => val + b[i][j]));
}

function subtractMatrix(a, b) {
  return a.map((row, i) => row.map((val, j) => val - b[i][j]));
}

function addVectorToMatrix(matrix, vector) {
  return matrix.map((row) => row.map((val, j) => val + vector[j]));
}

function relu(matrix) {
  return matrix.map((row) => row.map((val) => Math.max(0, val)));
}

function reluDerivative(matrix) {
  return matrix.map((row) => row.map((val) => (val > 0 ? 1 : 0)));
}

function softmax(matrix) {
  return matrix.map((row) => {
    const rowMax = Math.max(...row);
    const expValues = row.map((val) => Math.exp(val - rowMax));
    const expSum = expValues.reduce((a, b) => a + b);
    return expValues.map((val) => val / expSum);
  });
}

function hadamardProduct(a, b) {
  return a.map((row, i) => row.map((val, j) => val * b[i][j]));
}

function sumColumns(matrix) {
  return matrix[0].map((_, j) => matrix.reduce((sum, row) => sum + row[j], 0));
}

function generateData() {
  const X = createMatrix(BATCH_SIZE, INPUT_SIZE);
  const y = createMatrix(BATCH_SIZE, OUTPUT_SIZE, 0);
  for (let i = 0; i < BATCH_SIZE; i++) {
    const target = Math.floor(Math.random() * OUTPUT_SIZE);
    y[i][target] = 1;
  }
  return [X, y];
}

function forward(X, W1, b1, W2, b2) {
  const z1 = addVectorToMatrix(matrixMultiply(X, W1), b1);
  const a1 = relu(z1);
  const z2 = addVectorToMatrix(matrixMultiply(a1, W2), b2);
  const probs = softmax(z2);
  return [z1, a1, z2, probs];
}

function backward(X, y, probs, z1, a1, W1, W2) {
  const dz2 = subtractMatrix(probs, y);
  const dW2 = matrixMultiply(transpose(a1), dz2);
  const db2 = sumColumns(dz2);

  const da1 = matrixMultiply(dz2, transpose(W2));
  const dz1 = hadamardProduct(da1, reluDerivative(z1));
  const dW1 = matrixMultiply(transpose(X), dz1);
  const db1 = sumColumns(dz1);

  return [dW1, db1, dW2, db2];
}

function initWeights() {
  const scale1 = Math.sqrt(2.0 / INPUT_SIZE);
  const scale2 = Math.sqrt(2.0 / HIDDEN_SIZE);

  const W1 = createMatrix(INPUT_SIZE, HIDDEN_SIZE).map((row) => row.map((val) => val * scale1));
  const b1 = createVector(HIDDEN_SIZE, 0);
  const W2 = createMatrix(HIDDEN_SIZE, OUTPUT_SIZE).map((row) => row.map((val) => val * scale2));
  const b2 = createVector(OUTPUT_SIZE, 0);

  return [W1, b1, W2, b2];
}

function main() {
  const [W1, b1, W2, b2] = initWeights();
  const [X, y] = generateData();

  let forwardTime = 0;
  let backwardTime = 0;

  for (let i = 0; i < ITERATIONS; i++) {
    const startForward = performance.now();
    const [z1, a1, z2, probs] = forward(X, W1, b1, W2, b2);
    forwardTime += performance.now() - startForward;

    const startBackward = performance.now();
    const [dW1, db1, dW2, db2] = backward(X, y, probs, z1, a1, W1, W2);
    backwardTime += performance.now() - startBackward;
  }

  console.log('Размеры матриц:');
  console.log(`Батч: ${BATCH_SIZE}`);
  console.log(`Вход: ${INPUT_SIZE}`);
  console.log(`Скрытый слой: ${HIDDEN_SIZE}`);
  console.log(`Выход: ${OUTPUT_SIZE}`);
  console.log(`\nВремя выполнения (${ITERATIONS} итераций):`);
  console.log(`Forward pass: ${(forwardTime / ITERATIONS / 1000).toFixed(6)} сек`);
  console.log(`Backward pass: ${(backwardTime / ITERATIONS / 1000).toFixed(6)} сек`);
  console.log(`Общее время: ${((forwardTime + backwardTime) / ITERATIONS / 1000).toFixed(6)} сек`);
}

main();
