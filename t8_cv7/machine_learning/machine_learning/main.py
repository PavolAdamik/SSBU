from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from machine_learning.data_handling import Dataset
from machine_learning.experiment import Experiment
from machine_learning.result_plots import Plotter


def main():
    """
    Main function to execute the model training and evaluation pipeline.

    Initializes the dataset, defines models and their parameter grids,
    and invokes the replication of model training and evaluation.
    """
    # Initialize dataset and preprocess the data
    dataset = Dataset()

    # Define models to be trained
    models = {
        "Logistic Regression": LogisticRegression(solver='liblinear'),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=4, min_samples_split=3, min_samples_leaf=1)
    }

    # Define hyperparameter grids for tuning
    param_grids = {
        "Logistic Regression": {"C": [0.1, 1, 10], "max_iter": [10000]},
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 4, None],
            "min_samples_split": [3, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        }
    }

    experiment = Experiment(models, param_grids, n_replications=10)
    results = experiment.run(dataset.data, dataset.target)

    # Plot the results using the Plotter class
    plotter = Plotter()
    plotter.plot_metric_density(results)
    plotter.plot_evaluation_metric_over_replications(
        experiment.results.groupby('model')['accuracy'].apply(list).to_dict(),
        'Accuracy per Replication and Average Accuracy', 'Accuracy')
    plotter.plot_evaluation_metric_over_replications(
        experiment.results.groupby('model')['sensitivity'].apply(list).to_dict(),
        'Sensitivity per Replication and Average Sensitivity', 'Sensitivity')
    plotter.plot_confusion_matrices(experiment.mean_conf_matrices)
    plotter.print_best_parameters(results)


if __name__ == "__main__":
    main()
