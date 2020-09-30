import argparse


def args():
    """
    Function to create command line arguments

    Arguments:
        -dn <str> (data_name) name of the data to import form the data folder
            they are: breast-cancer, glass, iris, soybean, vote
        -rs <int> (random_seed) seed used for data split. Defaults to 1. All submitted output uses random_seed 1
        -kt <str> (knn_type) Define one of the specialized KNN functions. Can be edited or condensed. Default to regular
        -k <int> (k) K to replace tune function
        -s <float> (sigma) Sigma to replace tune function
        -e <float> (epsilon) Epsilon to replace tune function
    """
    # Initialize the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('-dn', '--data_name', help='Specify data name to extract and process')
    parser.add_argument('-rs', '--random_state', default=1, type=int,
                        help='Specify a seed to pass to the data splitter')
    parser.add_argument('-kt', '--knn_type', type=str,
                        help='Specify type of KNN, ignore for the original KNN, otherwise pass <edited> or <condensed>')
    parser.add_argument('-k', '--k', type=int,
                        help='Specify K for prediction. Overrides tune')
    parser.add_argument('-s', '--sigma', type=float,
                        help='Specify K for prediction. Overrides tune')
    parser.add_argument('-e', '--epsilon', type=float,
                        help='Specify epsilon for prediction. Overrides tune')

    # Parse arguments
    command_args = parser.parse_args()

    # Return the parsed arguments
    return command_args
