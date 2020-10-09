import argparse


def args():
    """
    Function to create command line arguments

    Arguments:
        -dn <str> (data_name) name of the data to import form the data folder
            they are: breast-cancer, car, segmentation, abalone, machine, forest-fires
        -rs <int> (random_seed) seed used for data split. Defaults to 1. All submitted output uses random_seed 1
        -p (prune) Trigger prune for classification tree. Does nothing for regressor
        -t (tune) Trigger tune for regression tree. Does nothing for classifier. This does not set thresholds, only
            output tune results
        -pt <float> (percent_threshold) Add a specific percentage threshold for early stopping on regressor. Does
            nothing for classifier
    """
    # Initialize the parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('-dn', '--data_name', help='Specify data name to extract and process')
    parser.add_argument('-rs', '--random_state', default=1, type=int,
                        help='Specify a seed to pass to the data splitter')
    parser.add_argument('-p', '--prune', action='store_true',
                        help='Trigger Prune on trained decision tree')
    parser.add_argument('-t', '--tune', action='store_true',
                        help='Trigger tune on CART decision tree')
    parser.add_argument('-pt', '--percent_threshold', default=0, type=float,
                        help='Specify percent_threshold for early stopping of tree')

    # Parse arguments
    command_args = parser.parse_args()

    # Return the parsed arguments
    return command_args
