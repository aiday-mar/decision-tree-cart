const Cart = require('../cart.js');

X = [
  [0, 1, 0],
  [0, 1, 1],
  [1, 1, 0],
  [1, 1, 1],
  [0, 0, 0],
  [0, 0, 1],
  [1, 0, 1],
  [1, 0, 0],
];

y = [0, 0, 0, 0, 0, 1, 1, 1];

let options = {
  seed: 3,
  maxFeatures: 2,
  replacement: false,
  nEstimators: 10,
  treeOptions: undefined,
  useSampleBagging: true,
};

let regression = new Cart.DecisionTreeRegression(options);
regression.train(X, y);
// indeed only one level created
console.log('The regression tree is : ', regression);
