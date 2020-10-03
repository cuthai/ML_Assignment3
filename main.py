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

    # Predict
    dt_model.predict()

    pass


if __name__ == '__main__':
    main()
