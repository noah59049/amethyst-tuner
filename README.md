# amethyst-tuner

This is the tuner specifically designed for the hand-crafted evaluation of [Amethyst](https://github.com/noah59049/Amethyst) using pytorch. The evaluation loss is minimized by gradient descent.

## Running
It is necessary to put `libamethyst_tuner.dylib` built from [`hcetuner.cpp`](https://github.com/noah59049/Amethyst/blob/master/hcetuner.cpp) in the main directory of the project.

It is also necessary to have a data file named `Noah.epd` in the main directory. I use a slightly modified version of [`lichess-big3-resolved.7z`](https://archive.org/details/lichess-big3-resolved.7z)

`tuner_functions.py` contains the functions needed. A sample training run is below, which is intended to be run from a Jupyter notebook that imports `tuner_functions`.

```python
import tuner_functions
tuner_functions.initialize_tensors() # Not strictly necessary, but better than doing it on the first epoch
run_data = {}
tuner = tuner_functions.get_model(
    optimizer = "adam",
    run_data = run_data
)
tuner_functions.print_continuous_weights(tuner)
tuner_functions.print_king_bucketed_psts(tuner)
save_eval_to_file(tuner, "NoahResults.txt")
```

This will print out the weights in a format designed for [`hce.cpp`](https://github.com/noah59049/Amethyst/blob/master/hce.cpp). `save_eval_to_file` saves the evals to a file in a format suited for the `runEvalTestSuite` test in [`tests.cpp`](https://github.com/noah59049/Amethyst/blob/master/tests.cpp).

`tuner_functions.py` also contains several tests, and running `python tuner_functions.py` will run the tests.

## Features
This tuner is, at its core, logistic regression. The eval weights are tuned to maximize the likelihood of the observed results. Some features supported include:
- King bucketed Piece Square Tables (PSTs), mobility evaluation.
- Cross entropy loss
- SGD, Adam, and LBFGS optimizers
- L2 regularization
- Variable learn rates
- Train-test split
- `nn.Embedding` optimizations for PSTs
- Graphing loss

## How it works
`libamethyst_tuner.dylib` contains functions for calculating piece-square tables and mobility that would be costly to calculate in Python, but are faster if implemented in C++. `tuner_functions.py` calls those functions from python. It also has functions for creating a tuner, running backpropagation, deciding when to stop training, and printing out eval weights.