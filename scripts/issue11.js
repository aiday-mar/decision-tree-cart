const Cart = require('../cart.js');

function approx(val, expected, eps) {
  return val - eps < expected && expected < val + eps;
}

let dataset = [
  [73, 'a', 'o', 152],
  [93, 'b', 'u', 185],
  [89, 'c', 'p', 180],
  [96, 'd', 'v', 196],
  [73, 'e', 'j', 142],
  [53, 'f', 'w', 101],
  [69, 'g', 'x', 149],
  [47, 'h', 'y', 115],
  [87, 'i', 'p', 175],
  [79, 'j', 'b', 164],
  [69, 'j', 'm', 141],
  [70, 'k', 'g', 141],
  [93, 'l', 'c', 184],
  [79, 'a', 'm', 152],
  [70, 'm', 'z', 148],
  [93, 'n', 'aa', 192],
  [78, 'o', 'ab', 147],
  [81, 'p', 'u', 183],
  [88, 'q', 's', 177],
  [78, 'r', 'x', 159],
  [82, 's', 'p', 177],
  [86, 't', 'n', 175],
  [78, 'r', 'ac', 175],
  [76, 'r', 'ad', 149],
  [96, 'u', 'l', 192],
];

let trainingSet = new Array(dataset.length);
let predictions = new Array(dataset.length);
let testingSet = new Array(dataset.length);

for (let i = 0; i < dataset.length; ++i) {
  trainingSet[i] = dataset[i].slice(0, 3);
  testingSet[i] = dataset[i].slice(0, 3);
  predictions[i] = dataset[i][3];
}

let options = {
  seed: 3,
  maxFeatures: 2,
  replacement: false,
  nEstimators: 10,
  treeOptions: undefined,
  useSampleBagging: true,
};

let regression = new Cart.DecisionTreeRegression(options);
regression.train(trainingSet, predictions);
let resultRegression = regression.predict(testingSet);
console.log('The result is: ', resultRegression);

let classifier = new Cart.DecisionTreeClassifier(options);
classifier.train(trainingSet, predictions);
let resultClassifier = classifier.predict(testingSet);
console.log('The result is: ', resultClassifier);
