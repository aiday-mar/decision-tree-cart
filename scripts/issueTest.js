const arrayMean = require('ml-array-mean');
const arrayMedian = require('ml-array-median');
const Matrix = require('ml-matrix').Matrix;
const WrapperMatrix2D = require('ml-matrix').WrapperMatrix2D;
const MatrixTransposeView = require('ml-matrix').MatrixTransposeView;
const MatrixColumnSelectionView =
  require('ml-matrix').MatrixColumnSelectionView;
const Cart = require('../cart.js');
const Random = require('random-js');

const selectionMethods = {
  mean: arrayMean,
  median: arrayMedian,
};

const defaultOptions = {
  maxFeatures: 1.0,
  replacement: false,
  nEstimators: 50,
  treeOptions: {},
  selectionMethod: 'mean',
  seed: 42,
  useSampleBagging: true,
  noOOB: false,
};

function approx(val, expected, eps) {
  return val - eps < expected && expected < val + eps;
}

// -- Utils --

function checkFloat(n) {
  return n > 0.0 && n <= 1.0;
}

/**
 * Select n with replacement elements on the training set and values, where n is the size of the training set.
 * @ignore
 * @param {Matrix} trainingSet
 * @param {Array} trainingValue
 * @param {number} seed - seed for the random selection, must be a 32-bit integer.
 * @return {object} with new X and y.
 */
function examplesBaggingWithReplacement(trainingSet, trainingValue, seed) {
  let engine;
  let distribution = Random.integer(0, trainingSet.rows - 1);
  if (seed === undefined) {
    engine = Random.MersenneTwister19937.autoSeed();
  } else if (Number.isInteger(seed)) {
    engine = Random.MersenneTwister19937.seed(seed);
  } else {
    throw new RangeError(
      `Expected seed must be undefined or integer not ${seed}`,
    );
  }

  let Xr = new Array(trainingSet.rows);
  let yr = new Array(trainingSet.rows);

  let oob = new Array(trainingSet.rows).fill(0);
  let oobN = trainingSet.rows;

  for (let i = 0; i < trainingSet.rows; ++i) {
    let index = distribution(engine);
    Xr[i] = trainingSet.getRow(index);
    yr[i] = trainingValue[index];

    if (oob[index]++ === 0) {
      oobN--;
    }
  }

  let Xoob = new Array(oobN);
  let ioob = new Array(oobN);

  // run backwards to have ioob filled in increasing order
  for (let i = trainingSet.rows - 1; i >= 0 && oobN > 0; --i) {
    if (oob[i] === 0) {
      Xoob[--oobN] = trainingSet.getRow(i);
      ioob[oobN] = i;
    }
  }

  return {
    X: new Matrix(Xr),
    y: yr,
    Xoob: new Matrix(Xoob),
    ioob,
    seed: engine.next(),
  };
}

/**
 * selects n features from the training set with or without replacement, returns the new training set and the indexes used.
 * @ignore
 * @param {Matrix} trainingSet
 * @param {number} n - features.
 * @param {boolean} replacement
 * @param {number} seed - seed for the random selection, must be a 32-bit integer.
 * @return {object}
 */
function featureBagging(trainingSet, n, replacement, seed) {
  if (trainingSet.columns < n) {
    throw new RangeError(
      'N should be less or equal to the number of columns of X',
    );
  }

  let distribution = Random.integer(0, trainingSet.columns - 1);
  let engine;
  if (seed === undefined) {
    engine = Random.MersenneTwister19937.autoSeed();
  } else if (Number.isInteger(seed)) {
    engine = Random.MersenneTwister19937.seed(seed);
  } else {
    throw new RangeError(
      `Expected seed must be undefined or integer not ${seed}`,
    );
  }

  let toRet = new Matrix(trainingSet.rows, n);

  let usedIndex;
  let index;
  if (replacement) {
    usedIndex = new Array(n);
    for (let i = 0; i < n; ++i) {
      index = distribution(engine);
      usedIndex[i] = index;
      toRet.setColumn(i, trainingSet.getColumn(index));
    }
  } else {
    usedIndex = new Set();
    index = distribution(engine);
    for (let i = 0; i < n; ++i) {
      while (usedIndex.has(index)) {
        index = distribution(engine);
      }
      toRet.setColumn(i, trainingSet.getColumn(index));
      usedIndex.add(index);
    }
    usedIndex = Array.from(usedIndex);
  }

  return {
    X: toRet,
    usedIndex: usedIndex,
    seed: engine.next(),
  };
}

/**
 * collects and combines the individual results from the tree predictions on Out-Of-Bag data
 * @ignore
 * @param {{index: {Array},predicted: {Array}}[]} oob: array of individual tree predictions
 * @param {array} y: true labels
 * @param {(predictions:{Array})=>{number}} aggregate: aggregation function
 * @return {Array}
 */
const collectOOB = (oob, y, aggregate) => {
  const res = Array(y.length);
  for (let i = 0; i < y.length; i++) {
    const all = [];
    for (let j = 0; j < oob.length; j++) {
      const o = oob[j];
      if (o.index[0] === i) {
        all.push(o.predicted[0]);
        o.index = o.index.slice(1);
        o.predicted = o.predicted.slice(1);
      }
    }
    res[i] = { true: y[i], all: all, predicted: aggregate(all) };
  }
  return res;
};

// -- End of Utils --

/**
 * @class RandomForestBase
 */
class RandomForestBase {
  /**
   * Create a new base random forest for a classifier or regression model.
   * @constructor
   * @param {object} options
   * @param {number|String} [options.maxFeatures] - the number of features used on each estimator.
   *        * if is an integer it selects maxFeatures elements over the sample features.
   *        * if is a float between (0, 1), it takes the percentage of features.
   * @param {boolean} [options.replacement] - use replacement over the sample features.
   * @param {number} [options.seed] - seed for feature and samples selection, must be a 32-bit integer.
   * @param {number} [options.nEstimators] - number of estimator to use.
   * @param {object} [options.treeOptions] - options for the tree classifier, see [ml-cart]{@link https://mljs.github.io/decision-tree-cart/}
   * @param {boolean} [options.isClassifier] - boolean to check if is a classifier or regression model (used by subclasses).
   * @param {boolean} [options.useSampleBagging] - use bagging over training samples.
   * @param {boolean} [options.noOOB] - don't calculate Out-Of-Bag predictions.
   * @param {object} model - for load purposes.
   */
  constructor(options, model) {
    if (options === true) {
      this.replacement = model.replacement;
      this.maxFeatures = model.maxFeatures;
      this.nEstimators = model.nEstimators;
      this.treeOptions = model.treeOptions;
      this.isClassifier = model.isClassifier;
      this.seed = model.seed;
      this.n = model.n;
      this.indexes = model.indexes;
      this.useSampleBagging = model.useSampleBagging;
      this.noOOB = true;
      this.maxSamples = model.maxSamples;

      let Estimator = this.isClassifier
        ? Cart.DecisionTreeClassifier
        : Cart.DecisionTreeRegression;
      this.estimators = model.estimators.map((est) => Estimator.load(est));
    } else {
      this.replacement = options.replacement;
      this.maxFeatures = options.maxFeatures;
      this.nEstimators = options.nEstimators;
      this.treeOptions = options.treeOptions;
      this.isClassifier = options.isClassifier;
      this.seed = options.seed;
      this.useSampleBagging = options.useSampleBagging;
      this.noOOB = options.noOOB;
      this.maxSamples = options.maxSamples;
    }
  }

  /**
   * Train the decision tree with the given training set and labels.
   * @param {Matrix|Array} trainingSet
   * @param {Array} trainingValues
   */
  train(trainingSet, trainingValues) {
    let currentSeed = this.seed;

    trainingSet = Matrix.checkMatrix(trainingSet);

    this.maxFeatures = this.maxFeatures || trainingSet.columns;
    this.numberFeatures = trainingSet.columns;
    this.numberSamples = trainingSet.rows;

    if (checkFloat(this.maxFeatures)) {
      this.n = Math.floor(trainingSet.columns * this.maxFeatures);
    } else if (Number.isInteger(this.maxFeatures)) {
      if (this.maxFeatures > trainingSet.columns) {
        throw new RangeError(
          `The maxFeatures parameter should be less than ${trainingSet.columns}`,
        );
      } else {
        this.n = this.maxFeatures;
      }
    } else {
      throw new RangeError(
        `Cannot process the maxFeatures parameter ${this.maxFeatures}`,
      );
    }

    if (this.maxSamples) {
      if (this.maxSamples < 0) {
        throw new RangeError(`Please choose a positive value for maxSamples`);
      } else {
        if (Utils.checkFloat(this.maxSamples)) {
          if (this.maxSamples > 1.0) {
            throw new RangeError(
              'Please choose either a float value between 0 and 1 or a positive integer for maxSamples',
            );
          } else {
            this.numberSamples = Math.floor(trainingSet.rows * this.maxSamples);
          }
        } else if (Number.isInteger(this.maxSamples)) {
          if (this.maxSamples > trainingSet.rows) {
            throw new RangeError(
              `The maxSamples parameter should be less than ${trainingSet.rows}`,
            );
          } else {
            this.numberSamples = this.maxSamples;
          }
        }
      }
    }

    if (this.maxSamples) {
      if (trainingSet.rows !== this.maxSamples) {
        let tmp = new Matrix(this.maxSamples, trainingSet.columns);
        for (let j = 0; j < this.maxSamples; j++) {
          tmp.removeRow(0);
        }
        for (let i = 0; i < this.maxSamples; i++) {
          tmp.addRow(trainingSet.getRow(i));
        }
        trainingSet = tmp;

        trainingValues = trainingValues.slice(0, this.maxSamples);
      }
    }

    let Estimator;
    if (this.isClassifier) {
      Estimator = Cart.DecisionTreeClassifier;
    } else {
      Estimator = Cart.DecisionTreeRegression;
    }

    this.estimators = new Array(this.nEstimators);
    this.indexes = new Array(this.nEstimators);

    let oobResults = new Array(this.nEstimators);

    for (let i = 0; i < this.nEstimators; ++i) {
      let res = this.useSampleBagging
        ? examplesBaggingWithReplacement(
            trainingSet,
            trainingValues,
            currentSeed,
          )
        : {
            X: trainingSet,
            y: trainingValues,
            seed: currentSeed,
            Xoob: undefined,
            yoob: [],
            ioob: [],
          };
      let X = res.X;
      let y = res.y;
      currentSeed = res.seed;
      let { Xoob, ioob } = res;

      res = featureBagging(X, this.n, this.replacement, currentSeed);
      X = res.X;
      currentSeed = res.seed;

      this.indexes[i] = res.usedIndex;
      this.estimators[i] = new Estimator(this.treeOptions);
      this.estimators[i].train(X, y);

      if (!this.noOOB && this.useSampleBagging) {
        let xoob = new MatrixColumnSelectionView(Xoob, this.indexes[i]);
        oobResults[i] = {
          index: ioob,
          predicted: this.estimators[i].predict(xoob),
        };
      }
    }
    if (!this.noOOB && this.useSampleBagging && oobResults.length > 0) {
      this.oobResults = collectOOB(
        oobResults,
        trainingValues,
        this.selection.bind(this),
      );
    }
  }

  printTrees() {
    for (let i = 0; i < this.nEstimators; ++i) {
      console.log('For Estimator : ', i, ' we have the following tree:');
      this.estimators[i].printTree();
    }
  }

  /**
   * Evaluate the feature importances for each tree in the ensemble
   * @return {Array} feature importances
   */
  featureImportance() {
    const trees = JSON.parse(JSON.stringify(this.estimators));
    const indexes = JSON.parse(JSON.stringify(this.indexes));
    let importance = [];

    function computeFeatureImportances(i, node) {
      // node.gain can be null or undefined
      if (!node || !('splitColumn' in node) || !(node.gain > 0)) return;
      let f = node.gain * node.numberSamples;
      if ('left' in node) {
        f -= (node.left.gain || 0) * (node.left.numberSamples || 0);
      }
      if ('right' in node) {
        f -= (node.right.gain || 0) * (node.right.numberSamples || 0);
      }
      importance[i][node.splitColumn] += f;
      if (node.left) {
        computeFeatureImportances(i, node.left);
      }
      if (node.right) {
        computeFeatureImportances(i, node.right);
      }
    }

    function normalizeImportances(i) {
      const s = importance[i].reduce((cum, v) => {
        return (cum += v);
      }, 0);
      importance[i] = importance[i].map((v) => {
        return v / s;
      });
    }

    for (let i = 0; i < trees.length; i++) {
      importance.push(new Array(this.numberFeatures).fill(0.0));
      computeFeatureImportances(i, trees[i].root);
      normalizeImportances(i);
    }

    let avgImportance = new Array(this.numberFeatures).fill(0.0);
    for (let i = 0; i < importance.length; i++) {
      for (let x = 0; x < this.numberFeatures; x++) {
        avgImportance[indexes[i][x]] += importance[i][x];
      }
    }

    const s = avgImportance.reduce((cum, v) => {
      return (cum += v);
    }, 0);
    return avgImportance.map((v) => {
      return v / s;
    });
  }

  /**
   * Method that returns the way the algorithm generates the predictions, for example, in classification
   * you can return the mode of all predictions retrieved by the trees, or in case of regression you can
   * use the mean or the median.
   * @abstract
   * @param {Array} values - predictions of the estimators.
   * @return {number} prediction.
   */
  // eslint-disable-next-line no-unused-vars
  selection(values) {
    throw new Error("Abstract method 'selection' not implemented!");
  }

  /**
   * Predicts the output given the matrix to predict.
   * @param {Matrix|Array} toPredict
   * @return {Array} predictions
   */
  predict(toPredict) {
    const predictionValues = this.predictionValues(toPredict);
    let predictions = new Array(predictionValues.rows);
    for (let i = 0; i < predictionValues.rows; ++i) {
      predictions[i] = this.selection(predictionValues.getRow(i));
    }

    return predictions;
  }

  /**
   * Predicts the output given the matrix to predict.
   * @param {Matrix|Array} toPredict
   * @return {MatrixTransposeView} predictions of estimators
   */
  predictionValues(toPredict) {
    let predictionValues = new Array(this.nEstimators);
    toPredict = Matrix.checkMatrix(toPredict);
    for (let i = 0; i < this.nEstimators; ++i) {
      let X = new MatrixColumnSelectionView(toPredict, this.indexes[i]);
      predictionValues[i] = this.estimators[i].predict(X);
    }
    return (predictionValues = new MatrixTransposeView(
      new WrapperMatrix2D(predictionValues),
    ));
  }

  /**
   * Returns the Out-Of-Bag predictions.
   * @return {Array} predictions
   */
  predictOOB() {
    if (!this.oobResults || this.oobResults.length === 0) {
      throw new Error(
        'No Out-Of-Bag results found. Did you forgot to train first?',
      );
    }
    return this.oobResults.map((v) => v.predicted);
  }

  /**
   * Export the current model to JSON.
   * @return {object} - Current model.
   */
  toJSON() {
    return {
      indexes: this.indexes,
      n: this.n,
      replacement: this.replacement,
      maxFeatures: this.maxFeatures,
      nEstimators: this.nEstimators,
      treeOptions: this.treeOptions,
      isClassifier: this.isClassifier,
      seed: this.seed,
      estimators: this.estimators.map((est) => est.toJSON()),
      useSampleBagging: this.useSampleBagging,
    };
  }
}

/**
 * @class RandomForestRegression
 * @augments RandomForestBase
 */
class RandomForestRegression extends RandomForestBase {
  /**
   * Create a new base random forest for a classifier or regression model.
   * @constructor
   * @param {object} options
   * @param {number} [options.maxFeatures=1.0] - the number of features used on each estimator.
   *        * if is an integer it selects maxFeatures elements over the sample features.
   *        * if is a float between (0, 1), it takes the percentage of features.
   * @param {boolean} [options.replacement=true] - use replacement over the sample features.
   * @param {number} [options.seed=42] - seed for feature and samples selection, must be a 32-bit integer.
   * @param {number} [options.nEstimators=50] - number of estimator to use.
   * @param {object} [options.treeOptions={}] - options for the tree classifier, see [ml-cart]{@link https://mljs.github.io/decision-tree-cart/}
   * @param {string} [options.selectionMethod="mean"] - the way to calculate the prediction from estimators, "mean" and "median" are supported.
   * @param {boolean} [options.useSampleBagging=true] - use bagging over training samples.
   * @param {number} [options.maxSamples=null] - if null, then draw X.shape[0] samples. If int, then draw maxSamples samples. If float, then draw maxSamples * X.shape[0] samples. Thus, maxSamples should be in the interval (0.0, 1.0].
   * @param {object} model - for load purposes.
   */
  constructor(options, model) {
    if (options === true) {
      super(true, model.baseModel);
      this.selectionMethod = model.selectionMethod;
    } else {
      options = Object.assign({}, defaultOptions, options);
      if (
        !(
          options.selectionMethod === 'mean' ||
          options.selectionMethod === 'median'
        )
      ) {
        throw new RangeError(
          `Unsupported selection method ${options.selectionMethod}`,
        );
      }

      options.isClassifier = false;

      super(options);
      this.selectionMethod = options.selectionMethod;
    }
  }

  /**
   * retrieve the prediction given the selection method.
   * @param {Array} values - predictions of the estimators.
   * @return {number} prediction
   */
  selection(values) {
    return selectionMethods[this.selectionMethod](values);
  }

  /**
   * Export the current model to JSON.
   * @return {object} - Current model.
   */
  toJSON() {
    let baseModel = super.toJSON();
    return {
      baseModel: baseModel,
      selectionMethod: this.selectionMethod,
      name: 'RFRegression',
    };
  }

  /**
   * Load a Decision tree classifier with the given model.
   * @param {object} model
   * @return {RandomForestRegression}
   */
  static load(model) {
    if (model.name !== 'RFRegression') {
      throw new RangeError(`Invalid model: ${model.name}`);
    }

    return new RandomForestRegression(true, model);
  }
}

let dataset = [
  [73, 80, 75, 152],
  [93, 88, 93, 185],
  [89, 91, 90, 180],
  [96, 98, 100, 196],
  [73, 66, 70, 142],
  [53, 46, 55, 101],
  [69, 74, 77, 149],
  [47, 56, 60, 115],
  [87, 79, 90, 175],
  [79, 70, 88, 164],
  [69, 70, 73, 141],
  [70, 65, 74, 141],
  [93, 95, 91, 184],
  [79, 80, 73, 152],
  [70, 73, 78, 148],
  [93, 89, 96, 192],
  [78, 75, 68, 147],
  [81, 90, 93, 183],
  [88, 92, 86, 177],
  [78, 83, 77, 159],
  [82, 86, 90, 177],
  [86, 82, 89, 175],
  [78, 83, 85, 175],
  [76, 83, 71, 149],
  [96, 93, 95, 192],
];

let trainingSet = new Array(dataset.length);
let predictions = new Array(dataset.length);

for (let i = 0; i < dataset.length; ++i) {
  trainingSet[i] = dataset[i].slice(0, 3);
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

let regression = new RandomForestRegression(options);
regression.train(trainingSet, predictions);
regression.printTrees();
let result = regression.predict(trainingSet);

console.log('The result is: ', result);
