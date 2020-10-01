from utils.args import args
from etl.etl import ETL
from decision_tree.id3_classifier import ID3Classifier


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
        'etl': etl
    }
    # KNN
    # Classification
    if arguments.data_name in ['breast-cancer', 'car', 'segmentation']:
        dt_model = ID3Classifier(**kwargs)
    # Regression
    else:
        pass

    # Fit
    dt_model.fit()

    # # Tune K and Sigma
    # dt_model.tune()
    #
    # # Tune and fit Epsilon for edited and condensed
    # if arguments.knn_type in ['edited', 'condensed']:
    #     # If epsilon was passed, used that
    #     if arguments.epsilon:
    #         epsilon_range = [arguments.epsilon]
    #     else:
    #         epsilon_range = None
    #
    #     # Tune and fit Epsilon
    #     dt_model.fit_modified(epsilon_range=epsilon_range)
    #
    # # If K and Sigma were passed, replace the tune outcome
    # if arguments.k:
    #     dt_model.k = arguments.k
    # if arguments.sigma:
    #     dt_model.sigma = arguments.sigma
    #
    # # Predict
    # dt_model.predict()
    #
    # # Output
    # dt_model.output()
    #
    # # Tune Visualize, if they weren't passed at the command line
    # if arguments.k == arguments.sigma == arguments.epsilon is None:
    #     dt_model.visualize_tune()


if __name__ == '__main__':
    main()
