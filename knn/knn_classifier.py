import pandas as pd
import json
import matplotlib.pyplot as plt
import numpy as np


class KNNClassifier:
    """
    Class KNNClassifier to handle classification of the data.

    This class implements a tune, predict, fit for edited and condensed, and output
    """
    def __init__(self, etl, knn_type):
        """
        Init function. Takes an ETL object as well as a KNN Type

        :param etl: etl, object with transformed and split data
        :param knn_type: str, specify for edited or condensed knn, leave blank for regular
        """
        # Set the attributes to hold our data
        self.etl = etl
        self.data_name = self.etl.data_name
        self.data_split = etl.data_split

        # Type of KNN, default to regular
        self.knn_type = 'regular'
        if knn_type:
            self.knn_type = knn_type

        # Train Data
        self.train_data = {}

        # Tune Results
        self.tune_results = {}
        self.k = 1

        # Test Results
        self.test_results = {}

        # Summary
        self.summary = {}
        self.summary_classification = None

    def fit(self):
        """
        Fit function

        This function doesn't actually do any fitting, it just specifies the train data for each CV split index. This
            combines the CV splits for each index. Ex: for split 0 CV 1-4 are combined and train_data 0 is set to CV 1-4
        """
        # Loop through index
        for index in range(5):
            # Remove the current index from the train_index
            train_index = [train_index for train_index in [0, 1, 2, 3, 4] if train_index != index]
            train_data = pd.DataFrame()

            # For index in our train_index, append to the Data Frame
            for data_split_index in train_index:
                train_data = train_data.append(self.data_split[data_split_index])

            # Update train data with the combined CV
            self.train_data.update({index: train_data})

    def tune(self, k_range=None):
        """
        Tune function

        this function tunes K by predicting the tune data split to each train_data (defined by the fit)

        :param k_range: list, range of Ks to tune, defaults to 3-21 skip 2
        """
        # Default K range
        if not k_range:
            k_range = list(range(3, 22, 2))

        # Specify tune variables
        tune_data = self.data_split['tune']
        tune_x = tune_data.iloc[:, :-1]

        # Set up dictionary to hold results
        tune_results = {k: [0, 0, 0, 0, 0] for k in k_range}

        # Loop through the 5 CV splits (train_data, so this is the 5 combined data sets)
        for index in range(5):
            # Define train variables
            train_data = self.train_data[index]
            train_x = train_data.iloc[:, :-1]

            # Go through each row of Tune
            for row_index, row in tune_x.iterrows():
                # Distance calculation and sorting
                distances = ((train_x - row) ** 2).sum(axis=1).sort_values()

                # Go through range of Ks
                for k in tune_results.keys():
                    # Specify neighbors
                    neighbors = distances[:k].index.to_list()

                    # Grab the classes of the neighbors
                    classes = train_data.loc[neighbors, 'Class']

                    # Grab the mode of the classes
                    class_occurrence = classes.mode()

                    # If there is more than one mode, take the class of the nearest neighbor
                    if len(class_occurrence) > 1:
                        classification = train_data.loc[neighbors[0], 'Class']
                    # Else just use the mode
                    else:
                        classification = class_occurrence[0]

                    # If the classification was wrong, add one to the result for misclassification
                    if classification != tune_data.loc[row_index, 'Class']:
                        tune_results[k][index] += 1

        # After looping, average the results for each K over all the CV splits
        for k in tune_results.keys():
            tune_results[k] = sum(tune_results[k]) / (len(tune_data) * 5)

        # Set the results and K
        self.tune_results = tune_results
        self.k = min(tune_results, key=tune_results.get)

    def fit_modified(self, epsilon_range=None):
        """
        Fit function for edited and condensed

        This implements both edited and condensed, which are smaller version of the original train data set. The call
            is made based on the KNN_Type passed
        """
        # Loop through each of the CV splits and edit or condense down that CV split
        for index in range(5):
            # Edited
            if self.knn_type == 'edited':
                self.edit(index)

            # Condensed
            else:
                self.condense(index)

    def edit(self, index, k=None):
        """
        Edit Function

        The function edits down the specified index using the edit logic. This is a recursive function that calls to
            itself if the edit_out_list is still > 0

        :param index: int, index of CV split to edit
        :param k: int, K, if non specified default to self.K, which is from the Tune function
        """
        # Grab K
        if not k:
            k = self.k

        # Train Data, based on Index passed, this function edits this so each recursive call pulls the edited data set
        train_data = self.train_data[index]
        train_x = train_data.iloc[:, :-1]

        # List for editing out
        edit_out_list = []

        # Loop through the train data
        for row_index, row in train_data.iterrows():
            # Distance calculation and sorting
            distances = ((train_x - row) ** 2).sum(axis=1).sort_values()[1:]

            # Define neighbors
            neighbors = distances[:k].index.to_list()

            # Grab neighbor classes
            classes = train_data.loc[neighbors, 'Class']

            # Grab the mode of the neighbors
            class_occurrence = classes.mode()

            # If there is more than one mode, use the class of nearest neighbor
            if len(class_occurrence) > 1:
                classification = train_data.loc[neighbors[0], 'Class']
            # Else use the mode
            else:
                classification = class_occurrence[0]

            # If misclassified, add to edit_out_list
            if classification != train_data.loc[row_index, 'Class']:
                edit_out_list.append(row_index)

        # Edit out the misclassified observations
        train_data = train_data.loc[~train_data.index.isin(edit_out_list)]

        # Update the train_data with our edited dataset
        self.train_data.update({index: train_data})

        # If there was an edit, recursive call to edit
        if len(edit_out_list) > 0:
            self.edit(index)

    def condense(self, index, z_data=None):
        """
        Condense Function

        The function condenses down the specified index using the condense logic. This is a recursive function that
            calls to itself if the condense_in_list is still > 0

        :param index: int, index of CV split to edit
        :param z_data: DataFrame, current condensed data set
        """
        # Grab the train data
        temp_train_data = self.train_data[index]

        # If z was not passed, initialize z with the first result of the train set
        if z_data is None:
            z_data = pd.DataFrame(temp_train_data.iloc[0, :]).T
        z_data_x = z_data.iloc[:, :-1]

        # Variable for holding in values to condense in
        condense_in_count = 0

        # Loop through our train data set
        for row_index, row in temp_train_data.iterrows():
            # Calculate distances
            distances = ((z_data_x - row) ** 2).sum(axis=1).sort_values()

            # Define neighbors
            neighbor = distances.index.to_list()[0]

            # Grab neighbor classes
            classification = z_data.loc[neighbor, 'Class']

            # If misclassification, add to condense, and update Z
            if classification != temp_train_data.loc[row_index, 'Class']:
                condense_in_count += 1
                z_data = z_data.append(temp_train_data.loc[row_index])
                z_data_x = z_data.iloc[:, :-1]

        # If condense occurred, recursively call to condense
        if condense_in_count > 0:
            self.condense(index, z_data)
        # Else update our train data with Z
        else:
            self.train_data.update({index: z_data})

    def predict(self, k=None):
        """
        Predict function

        Function to predict on our test data set

        :param k: int, K, if non specified default to self.K, which is from the Tune function
        """
        # Define K
        if not k:
            k = self.k

        # Set up dictionary to hold end results
        test_results = {
            index: {
                'misclassification': 0,
                'classification': []
            } for index in range(5)
        }

        # Loop through each of the test splits
        for index in range(5):
            # Define train
            train_data = self.train_data[index]
            train_x = train_data.iloc[:, :-1]

            # Define test
            test_data = self.data_split[index]
            test_x = test_data.iloc[:, :-1]

            # Loop through test data set
            for row_index, row in test_x.iterrows():
                # Calculate distances
                distances = ((train_x - row) ** 2).sum(axis=1).sort_values()

                # Define neighbors
                neighbors = distances[:k].index.to_list()

                # Grab neighbor classes
                classes = train_data.loc[neighbors, 'Class']

                # Grab the mode of the classes
                class_occurrence = classes.mode()

                # If more than one mode, use the class of the nearest neighbor
                if len(class_occurrence) > 1:
                    classification = train_data.loc[neighbors[0], 'Class']
                # Else use the mode
                else:
                    classification = class_occurrence[0]

                # If misclassification, add to results
                if classification != test_data.loc[row_index, 'Class']:
                    test_results[index]['misclassification'] += 1

                # Append classification to results
                test_results[index]['classification'].append(classification)

            # Calculate misclassification error
            test_results[index]['misclassification'] = test_results[index]['misclassification'] / len(test_data)

        # Set test results
        self.test_results = test_results

    def output(self):
        """
        Output function to output results

        :return JSON: summary of tune and train
        :return CSV: csv of classification over train data sets
        """
        # Calculate misclassification
        misclassification = sum([self.test_results[index]['misclassification'] for index in range(5)])

        # Summary JSON
        self.summary = {
            'tune': {
                'k': self.k
            },
            'test': {
                'misclassification': misclassification / 5
            }
        }

        # Output JSON
        with open(f'output_{self.data_name}\\{self.data_name}_{self.knn_type}_summary.json', 'w') as file:
            json.dump(self.summary, file)

        # Summary CSV
        summary_classification = pd.DataFrame()

        # Loop through each test data set and add the results
        for index in range(5):
            temp_summary_classification = self.data_split[index]
            temp_summary_classification['classification'] = self.test_results[index]['classification']

            # Append temp to the CSV
            summary_classification = summary_classification.append(temp_summary_classification)

        # Dump CSV and save
        summary_classification.to_csv(f'output_{self.data_name}\\{self.data_name}_{self.knn_type}_classification.csv')
        self.summary_classification = summary_classification

    def visualize_tune(self):
        """
        Tune visualization function

        This function uses the results of the tune function to create a plot graph

        :return: matplotlib saved jpg in output folder
        """
        # Figure / axis set up
        fig, ax = plt.subplots()

        # We'll plot the list of params and their accuracy
        ax.plot(self.tune_results.keys(), self.tune_results.values())

        # Title
        ax.set_title(rf'{self.data_name} Tune Results - Optimal: K {self.k}')

        # X axis
        ax.set_xlabel('K')
        ax.set_xlim(3, 21)
        ax.set_xticks(np.linspace(3, 21, 10))
        ax.set_xticklabels(np.linspace(3, 21, 10), rotation=45, fontsize=6)

        # Y axis
        ax.set_ylabel('Misclassification')

        # Saving
        plt.savefig(f'output_{self.data_name}\\{self.data_name}_tune.jpg')
