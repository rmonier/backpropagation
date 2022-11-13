# Backpropagation Neural Network & MLR

Note: You can see the README of the subdirectories for more details about them.

## Prerequisites

- Install [Python 3.11](https://www.python.org/downloads/release/python-3110/)
- Install [Pipenv](https://pipenv.pypa.io/): `pip install pipenv`
- Install [Julia 1.8](https://julialang.org/downloads/). **It has to be available in the path.**

## Installation

Install the dependencies by doing:
```sh
pipenv install --dev
pipenv run install
```

## Usage

Run a script by doing:
```sh
pipenv run <script>
```

### Example of workflow execution

1. Preprocess the raw data:
    ```sh
    pipenv run preprocess
    ```
2. Normalize the preprocessed data:
    ```sh
    pipenv run normalize
    ```
3. Split the normalized data into training and test sets:
    ```sh
    pipenv run split
    ```
4. Analyze the sets to find the best parameters:
    ```sh
    pipenv run analyze
    ```
5. Train the MLR models:
    ```sh
    pipenv run train_mlr
    ```
6. Evaluate the models:
    ```sh
    pipenv run evaluate
    ```
7. You can now edit the Pipfile to set the found best parameters (already done), and train the models with them:
    ```sh
    pipenv run train_bp_turbine
    pipenv run train_bp_synthetic
    pipenv run train_bp_top10s
    ```

Note: During development, when adding or removing a package to the Julia project, regenerate the manifest with `julia --project=. -e 'using Pkg; Pkg.resolve()'`

## Credits

Romain Monier [ [GitHub](https://github.com/rmonier) ] – Co-developer

Marlon Funk [ [GitHub](https://github.com/MarlonFunk) ] – Co-developer