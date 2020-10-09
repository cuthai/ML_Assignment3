from utils.args import args
from etl.etl import ETL
from decision_tree.id3_classifier import ID3Classifier
from decision_tree.cart_regressor import CARTRegressor


def main():
    # Parse arguments
    arguments = args()

    # Set up kwargs for ETL
    kwargs = {
        'data_name': arguments.data_name,
        'random_state': arguments.random_state
    }
    etl = ETL(**kwargs)

    # KNN
    # Classification
    if arguments.data_name in ['breast-cancer', 'car', 'segmentation']:
        # Set up kwargs for KNN
        kwargs = {
            'etl': etl,
            'prune': arguments.prune
        }

        dt_model = ID3Classifier(**kwargs)
    # Regression
    else:
        # Set up kwargs for KNN
        kwargs = {
            'etl': etl,
            'percent_threshold': arguments.percent_threshold
        }

        dt_model = CARTRegressor(**kwargs)

        if arguments.tune:
            dt_model.tune()

    # Fit
    dt_model.fit()

    # Predict
    dt_model.predict()

    # Summarize
    dt_model.summarize()


if __name__ == '__main__':
    main()
