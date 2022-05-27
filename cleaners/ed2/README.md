# Example-Driven Error Detection

Traditional error detection approaches require user-defined parameters and rules. Thus, the user has to know both the error detection system and the data. However, we can also formulate error detection as a semi-supervised classification problem that only requires domain expertise. The challenges for such an approach are twofold: (1) to represent the data in a way that enables a classification model to identify various kinds of data errors, and (2) to pick the most promising data values for learning. In this paper, we address these challenges with our new example-driven error detection method (ED2). First, we discuss and identify the appropriate features to locate different kinds of data errors across different data types. Second, we present a new two-dimensional multi-classifier sampling strategy for active learning. The combined application of these techniques enables the convergence of the classification task with high detection accuracy. On several real-world datasets, ED2 requires, on average, only 1\% labels to outperform existing error detection approaches that are manually configured and tuned.

## Datasets
We provide the dirty and the clean version of a number of [datasets](../master/datasets).

## Additional Evaluations
In addition to the charts provided in the paper, we provide additional evaluations on more datasets:

1) [Feature representations](../master/documentation/evaluations/features.pdf): Besides more datasets, we also provide the F1-score for LSTM features on Address, Flights, and Hospital.
2) [Column selection strategies](../master/documentation/evaluations/column_selection.pdf)
3) [Classification models](../master/documentation/evaluations/models.pdf)

## Documentation
We are working hard to provide as much documentation as possible over the time. We start here:
1) [Constraints that we used to run NADEEF](../master/documentation/NADEEF_DCs.md)


## Using ED2
To run the experiments, first, you need to set the paths in a configuration file with the name of your machine. Examples can be found here: ~/model/ml/configuration/resources/

Then, you can adapt the file ~/model/ml/experiments/features_experiment_multi.py to run the experiments that you are interested in.


## Scenario
![ed2 dashboard](https://user-images.githubusercontent.com/5217389/47003848-77012800-d130-11e8-8765-5e9f9e3f0010.png)
