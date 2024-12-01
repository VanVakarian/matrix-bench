const tf = require('@tensorflow/tfjs-node');

// const BATCH_SIZE = 1000;
// const INPUT_SIZE = 784;
// const HIDDEN_SIZE = 512;
// const OUTPUT_SIZE = 10;
// const ITERATIONS = 50;

const BATCH_SIZE = 2150;
const INPUT_SIZE = 1686;
const HIDDEN_SIZE = 1100;
const OUTPUT_SIZE = 10;
const ITERATIONS = 50;

async function generateData() {
  const X = tf.randomNormal([BATCH_SIZE, INPUT_SIZE]);
  const y = tf.oneHot(tf.randomUniform([BATCH_SIZE], 0, OUTPUT_SIZE, 'int32'), OUTPUT_SIZE);
  return [X, y];
}

async function initWeights() {
  const scale1 = Math.sqrt(2.0 / INPUT_SIZE);
  const scale2 = Math.sqrt(2.0 / HIDDEN_SIZE);

  const W1 = tf.randomNormal([INPUT_SIZE, HIDDEN_SIZE]).mul(scale1);
  const b1 = tf.zeros([HIDDEN_SIZE]);
  const W2 = tf.randomNormal([HIDDEN_SIZE, OUTPUT_SIZE]).mul(scale2);
  const b2 = tf.zeros([OUTPUT_SIZE]);

  return [W1, b1, W2, b2];
}

async function forward(X, W1, b1, W2, b2) {
  const z1 = tf.add(tf.matMul(X, W1), b1);
  const a1 = tf.relu(z1);
  const z2 = tf.add(tf.matMul(a1, W2), b2);
  const probs = tf.softmax(z2);
  return [z1, a1, z2, probs];
}

async function backward(X, y, probs, z1, a1, W1, W2) {
  const dz2 = tf.sub(probs, y);
  const dW2 = tf.matMul(a1.transpose(), dz2);
  const db2 = tf.sum(dz2, [0]);

  const da1 = tf.matMul(dz2, W2.transpose());
  const dz1 = tf.mul(da1, tf.step(z1));
  const dW1 = tf.matMul(X.transpose(), dz1);
  const db1 = tf.sum(dz1, [0]);

  return [dW1, db1, dW2, db2];
}

async function main() {
  const [W1, b1, W2, b2] = await initWeights();
  const [X, y] = await generateData();

  let forwardTime = 0;
  let backwardTime = 0;

  for (let i = 0; i < ITERATIONS; i++) {
    const startForward = performance.now();
    const [z1, a1, z2, probs] = await forward(X, W1, b1, W2, b2);
    forwardTime += performance.now() - startForward;

    const startBackward = performance.now();
    const [dW1, db1, dW2, db2] = await backward(X, y, probs, z1, a1, W1, W2);
    backwardTime += performance.now() - startBackward;

    tf.dispose([z1, a1, z2, probs, dW1, db1, dW2, db2]);
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

  tf.dispose([W1, b1, W2, b2, X, y]);
}

main().catch(console.error);
