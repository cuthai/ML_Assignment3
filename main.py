from utils.args import args
from etl.etl import ETL


def main():
    # Parse arguments
    arguments = args()

    # Set up kwargs for ETL
    kwargs = {
        'data_name': arguments.data_name,
        'random_state': arguments.random_state
    }
    etl = ETL(**kwargs)

    # Set up kwargs for KNN
    kwargs = {
        'etl': etl,
        'knn_type': arguments.knn_type
    }
    # KNN
    # Classification
    if arguments.data_name in ['glass', 'segmentation', 'vote']:
        knn_model = KNNClassifier(**kwargs)
    # Regression
    else:
        knn_model = KNNRegressor(**kwargs)

    # Fit
    knn_model.fit()

    # Tune K and Sigma
    knn_model.tune()

    # Tune and fit Epsilon for edited and condensed
    if arguments.knn_type in ['edited', 'condensed']:
        # If epsilon was passed, used that
        if arguments.epsilon:
            epsilon_range = [arguments.epsilon]
        else:
            epsilon_range = None

        # Tune and fit Epsilon
        knn_model.fit_modified(epsilon_range=epsilon_range)

    # If K and Sigma were passed, replace the tune outcome
    if arguments.k:
        knn_model.k = arguments.k
    if arguments.sigma:
        knn_model.sigma = arguments.sigma

    # Predict
    knn_model.predict()

    # Output
    knn_model.output()

    # Tune Visualize, if they weren't passed at the command line
    if arguments.k == arguments.sigma == arguments.epsilon is None:
        knn_model.visualize_tune()


if __name__ == '__main__':
    main()
