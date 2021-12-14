# Task 1: Evaluator
> We want to write some tooling to help evaluate multiclass classifiers. 
> So far we have a `ConfusionMatrix` class in `evaluator_utils.py`, but we need to write an `Evaluator` that populates it with values and calculates some metrics.

The following files are relevant to this first task:

* `README.md` this doc 😉
* `evaluator.py` containing the implementation of the `Evaluator` class. This will be where you write your solution and you can execute it to test the code.
* `evaluator_utils.py` some helper types, functions and constants.

## Subtasks

1. As a warmup, implement `Evaluator.build_matrix()` in `evaluator.py`.
2. Implement `Evaluator.calculate_metrics()` in `evaluator.py`.

## Running the code

```shell
python3 evaluator.py
```

The code should print output like:
```
🍌	🍎	🥑	⬅️  predictions / ⬇️  actuals
26	1	0	🍌
1	24	1	🍎
0	0	47	🥑
{'f1': 0.9629629629629629,
 'label': '🍌',
 'num_actual_samples': 27,
 'precision': 0.9629629629629629,
 'recall': 0.9629629629629629}
{'f1': 0.9411764705882353,
 'label': '🍎',
 'num_actual_samples': 26,
 'precision': 0.96,
 'recall': 0.9230769230769231}
{'f1': 0.9894736842105264,
 'label': '🥑',
 'num_actual_samples': 47,
 'precision': 0.9791666666666666,
 'recall': 1.0}
```

# Task 2: Clustering
> Next, we would like to implement an algorithm to cluster a dataset of embeddings in some way - how you choose to do this is mostly up to you! 
> To give the problem some context, let's imagine that the embeddings provided are word embeddings, and we are trying to determine if groups of words have similar meanings.
> To do this we want to be able to assign a cluster ID to each embedding that is similar. 
>
> To start, we have a `ClusteringModel` class in `clustering.py`; we need to implement the `fit()` function so that the model takes a list of embeddings as an input, and returns a dictionary that maps a cluster ID to a list of the embeddings that have been assigned to that cluster.

The following files are relevant to this task:

* `README.md` this doc 😉
* `clustering.py` containing the implementation of the `ClusteringModel` class. This will be where you write your solution and you can execute it to test the code.
* `clustering_utils.py` some helper types, functions and constants.

## Running the code
```shell
python3 clustering.py
```

The code should print output like:
```
Precision 0.63	Recall 0.88	F1 0.74
```

The dataset is randomly generated, but a working solution should be able to score in the 70s for F1 averaged over a few runs.