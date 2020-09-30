import pandas as pd
import numpy as np
import copy
import json
import matplotlib.pyplot as plt


class KNNRegressor:
    """
    Class KNNRegressor to handle prediction of the data.

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
        self.sigma = 1
        self.epsilon = .05

        # Test Results
        self.test_results = {}

        # Summary
        self.summary = {}
        self.summary_prediction = None

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

            # For index in our train_index, append to the Data Frame
            train_data = pd.DataFrame()
            for data_split_index in train_index:
                train_data = train_data.append(self.data_split[data_split_index])

            # Update train data with the combined CV
            self.train_data.update({index: train_data})

    def tune(self, k_range=None, k=None, sigma_range=None, sigma=None):
        """
        Tune function

        This function tunes k and Sigma by calling to the specific function

        :param k_range: list, range of Ks to tune
        :param k: int, k, used with sigma_range to tune just sigma
        :param sigma_range: list, range of sigma to tune
        :param sigma: float, sigma, used with k_range to tune just k
        """
        # k
        self.tune_k(k_range, sigma)

        # Sigma
        self.tune_sigma(k, sigma_range)

    def tune_k(self, k_range=None, sigma=None):
        """
        Tune function for K

        This function tunes K and uses a default sigma = 1

        :param k_range: list, range of Ks to tune, defaults to 3-21 skip 2
        :param sigma: float, sigma, used with k_range to tune just k
        """
        # Default K range
        if not k_range:
            k_range = list(range(3, 22, 2))

        # Default sigma, if none this will be set to 1
        if not sigma:
            sigma = self.sigma

        # Define tune variables
        tune_data = self.data_split['tune']
        tune_x = tune_data.iloc[:, :-1]
        tune_y = tune_data.iloc[:, -1]

        # Dictionary to hold results
        tune_results = {'k': {k: [0, 0, 0, 0, 0] for k in k_range}}

        # Loop through each of the 5 CV splits
        for index in range(5):
            # Grab train variables
            train_data = self.train_data[index]
            train_x = train_data.iloc[:, :-1]
            train_y = train_data.iloc[:, -1]

            # Go through the tune data set
            for row_index, row in tune_x.iterrows():
                # Distance calculation and sorting
                distances = ((train_x - row) ** 2).sum(axis=1).sort_values()

                # Go through each K
                for k in tune_results['k'].keys():
                    # Grab the neighbors
                    neighbors = distances[:k]

                    # Calculate neighbor kernels
                    kernel = np.exp((1 / (2 * sigma)) * neighbors)

                    # Grab actual value of neighbors
                    neighbors_r = train_y.loc[neighbors.index.to_list()]

                    # Prediction calculation
                    prediction = sum(kernel * neighbors_r) / sum(kernel)

                    # Grab actual for comparison
                    actual = tune_y.loc[row_index]

                    # Add squared error to results
                    tune_results['k'][k][index] += (actual - prediction) ** 2

        # For each K, average the squared error to MSE
        for k in tune_results['k'].keys():
            tune_results['k'][k] = sum(tune_results['k'][k]) / (len(tune_data) * 5)

        # Update results, set K
        self.tune_results.update(tune_results)
        self.k = min(tune_results['k'], key=tune_results['k'].get)

    def tune_sigma(self, k=None, sigma_range=None):
        """
        Tune function for sigma

        This function tunes sigma after K, so K is set from the tune_k function

        :param k: int, k, used with sigma_range to tune just sigma
        :param sigma_range: list, range of sigma to tune, defaults to .5 - 3 skip .5
        """
        # Grab default K, after tuning K
        if not k:
            k = self.k

        # Default sigma range
        if not sigma_range:
            sigma_range = np.linspace(.5, 3, 6)

        # Define tune variables
        tune_data = self.data_split['tune']
        tune_x = tune_data.iloc[:, :-1]
        tune_y = tune_data.iloc[:, -1]

        # Dictionary to hold results
        tune_results = {'sigma': {sigma: [0, 0, 0, 0, 0] for sigma in sigma_range}}

        # Loop through each of the 5 CV splits
        for index in range(5):
            # Grab train variables
            train_data = self.train_data[index]
            train_x = train_data.iloc[:, :-1]
            train_y = train_data.iloc[:, -1]

            # Go through the tune data set
            for row_index, row in tune_x.iterrows():
                # Distance calculation and sorting
                distances = ((train_x - row) ** 2).sum(axis=1).sort_values()

                # Go through each sigma
                for sigma in tune_results['sigma'].keys():
                    # Grab neighbors
                    neighbors = distances[:k]

                    # Kernel calculation
                    kernel = np.exp((1 / (2 * sigma)) * neighbors)

                    # Grab actual neighbor values
                    neighbors_r = train_y.loc[neighbors.index.to_list()]

                    # Prediction calculation
                    prediction = sum(kernel * neighbors_r) / sum(kernel)

                    # Grab actual for comparison
                    actual = tune_y.loc[row_index]

                    # Add squared error to results
                    tune_results['sigma'][sigma][index] += (actual - prediction) ** 2

        # For each sigma, average the squared error to MSE
        for sigma in tune_results['sigma'].keys():
            tune_results['sigma'][sigma] = sum(tune_results['sigma'][sigma]) / (len(tune_data) * 5)

        # Update results, set Sigma
        self.tune_results.update(tune_results)
        self.sigma = min(tune_results['sigma'], key=tune_results['sigma'].get)

    def fit_modified(self, epsilon_range=None):
        """
        Function to fit edited or condensed data set

        This function first tunes the epsilon and then choses the best result as the finished edited or condensed data
            set. This function uses the already tuned K and Sigma.

        :param epsilon_range: list, range of epsilon to tune, defaults from .5 to .01
        """
        # Default epsilon range
        if not epsilon_range:
            epsilon_range = [.5, .25, .1, .05, .01]

        # Define a really high starting MSE to beat
        min_mse = 1000000

        # Hold results for each epsilon
        temp_train_data = {epsilon: {} for epsilon in epsilon_range}
        min_train_data = None

        # Initialize epsilon in the tune results
        self.tune_results['epsilon'] = {}

        # Loop through each epsilon
        for epsilon in epsilon_range:
            # Initial MSE of 0
            epsilon_mse = 0

            # Loop through each CV split
            for index in range(5):
                # edited
                if self.knn_type == 'edited':
                    # Get final edited train data and MSE
                    train_data, mse = self.edit(copy.deepcopy(self.train_data[index]), epsilon=epsilon)

                    # Add MSE for this epsilon
                    epsilon_mse += mse

                    # Add final edited train_data to the temporary dictionary for this epsilon
                    temp_train_data[epsilon].update({index: train_data})

                # condensed
                else:
                    # Get final condensed train data and MSE
                    train_data, mse = self.condense(copy.deepcopy(self.train_data[index]), epsilon=epsilon)

                    # Add MSE for this epsilon
                    epsilon_mse += mse

                    # Add final edited train_data to the temporary dictionary for this epsilon
                    temp_train_data[epsilon].update({index: train_data})

            # Once all 5 CV splits are done, average the MSE
            average_mse = epsilon_mse / 5

            # If this epsilon's average MSE is lower than the current min,
            if average_mse < min_mse:
                # Set min MSE to current
                min_mse = average_mse

                # Set the best train_data to current
                min_train_data = temp_train_data[epsilon]

                # Set self epsilon to current
                self.epsilon = epsilon

            # Update tune results with this epsilon's results
            self.tune_results['epsilon'].update({epsilon: average_mse})

        # Set our train_data to the best performing data set
        self.train_data = min_train_data

    def edit(self, temp_train_data, k=None, sigma=None, epsilon=None):
        """
        Edit Function

        This is a recursive function that passes the train_data to itself to continue to edit until no more edits are
            made. At the end MSE is calculated by calling to tune_epsilon which calculates MSE against edited data and
            tune

        :param temp_train_data: DataFrame, starts with the full train data, passes edited data to itself
        :param k: int, K, defaults to tuned K
        :param sigma: float, sigma, defaults to tuned sigma
        :param epsilon: float, epsilon, from the tune function
        :return train_data: final edited data set
        :return mse: mse of the edited data against tune data set
        """
        # Stop case if the data is too small to edit
        if len(temp_train_data) <= 1:
            return temp_train_data, self.tune_epsilon(temp_train_data)

        # Default k
        if not k:
            k = self.k

        # Default sigma
        if not sigma:
            sigma = self.sigma

        # Default epsilon, should be passed
        if not epsilon:
            epsilon = self.epsilon

        # Grab train data
        train_data = temp_train_data
        train_x = train_data.iloc[:, :-1]
        train_y = train_data.iloc[:, -1]

        # Initialize edit out list
        edit_out_list = []

        # Loop through train data
        for row_index, row in train_data.iterrows():
            # Calculate distances
            distances = ((train_x - row) ** 2).sum(axis=1).sort_values()[1:]

            # Grab neighbors
            neighbors = distances[:k]

            # Kernel calculation
            kernel = np.exp((1 / (2 * sigma)) * neighbors)

            # Grab actual values of neighbors
            neighbors_r = train_y.loc[neighbors.index.to_list()]

            # Calculate prediction
            prediction = sum(kernel * neighbors_r) / sum(kernel)

            # Grab the actual for comparison
            actual = train_y.loc[row_index]

            # If the actual is 0 set the percent different to 1
            # This is for the forest-fires data set
            if actual == 0:
                percent_different = 1
            # Else calculate the percentage difference from prediction to actual
            else:
                percent_different = abs((prediction - actual) / actual)

            # If the percent is higher than epsilon, add to edit out list
            if percent_different > epsilon:
                edit_out_list.append(row_index)

        # Edit out data, in batches
        train_data = train_data.loc[~train_data.index.isin(edit_out_list)]

        # Stop case if the train data has edited itself to 0
        if len(train_data) == 0:
            return train_data, 1000001

        # Else if there were edits, recursive call
        if len(edit_out_list) > 0:
            train_data, mse = self.edit(train_data, epsilon=epsilon)

        # If no edits were made, calculate MSE
        else:
            mse = self.tune_epsilon(train_data)

        # Return the final edit and MSE after all edits done
        return train_data, mse

    def condense(self, temp_train_data, sigma=None, epsilon=None, z_data=None):
        """
        Condense Function

        This is a recursive function that passes the z_data to itself to continue to condense until no more condense are
           made. At the end MSE is calculated by calling to tune_epsilon which calculates MSE against condense data and
           tune

        :param temp_train_data: DataFrame, starts with the full train data
        :param sigma: float, sigma, defaults to tuned sigma
        :param epsilon: float, epsilon, from the tune function
        :param z_data: DataFrame, current condensed DataFrame, passes to the next, initial is the first row in train
        :return train_data: final edited data set
        :return mse: mse of the edited data against tune data set
        """
        # Default sigma
        if not sigma:
            sigma = self.sigma

        # Default epsilon, should be passed
        if not epsilon:
            epsilon = self.epsilon

        # Initial z_data, set to the first row if none
        if z_data is None:
            z_data = pd.DataFrame(temp_train_data.iloc[0, :]).T
        z_data_x = z_data.iloc[:, :-1]
        z_data_y = z_data.iloc[:, -1]

        # Initialize condense in list
        condense_in_count = 0

        # Remove any variables already condensed out of the train data
        temp_train_data = temp_train_data.loc[~temp_train_data.index.isin(z_data.index)]

        # Loop through train data
        for row_index, row in temp_train_data.iterrows():
            # Calculate distances
            distances = ((z_data_x - row) ** 2).sum(axis=1).sort_values()

            # Grab closest neighbor
            neighbors = distances[:1]

            # Kernel calculation
            kernel = np.exp((1 / (2 * sigma)) * neighbors)

            # Grab actual value of neighbor
            neighbors_r = z_data_y.loc[neighbors.index.to_list()]

            # Calculate prediction
            prediction = sum(kernel * neighbors_r) / sum(kernel)

            # Grab the actual for comparison
            actual = temp_train_data.loc[row_index][-1]

            # If the actual is 0 set the percent different to 1 if the prediction is different from actual
            # This is for the forest-fires data set
            if actual == 0:
                # If different set to 1
                if prediction > 0:
                    percent_different = 1
                # If no different set to 0
                else:
                    percent_different = 0
            # Else calculate the percentage difference from prediction to actual
            else:
                percent_different = abs((prediction - actual) / actual)

            # If the percent is higher than epsilon, add to condense in
            if percent_different > epsilon:
                condense_in_count += 1
                # Redefine the z_data
                z_data = z_data.append(temp_train_data.loc[row_index])
                z_data_x = z_data.iloc[:, :-1]
                z_data_y = z_data.iloc[:, -1]

        # If there were condensed observations, call to condense again
        if condense_in_count > 0:
            z_data, mse = self.condense(temp_train_data, epsilon=epsilon, z_data=z_data)

        # If no condensed, calculate MSE
        else:
            mse = self.tune_epsilon(z_data)

        # Return the final z_data and MSE
        return z_data, mse

    def tune_epsilon(self, train_data, k=None, sigma=None):
        """
        Function to tune epsilon

        This function returns a MSE comparing the Tune data to the passed edited or condensed data set

        :param train_data: DataFrame, final edited or condensed data set
        :param k: int, K, defaults to tuned k
        :param sigma: float, sigma, defaults to tuned sigma
        :return MSE: float, MSE calculation of predictions on tuned using train
        """
        # Default K
        if not k:
            k = self.k

        # Default sigma
        if not sigma:
            sigma = self.sigma

        # Grab tune variables
        tune_data = self.data_split['tune']
        tune_x = tune_data.iloc[:, :-1]
        tune_y = tune_data.iloc[:, -1]

        # Grab train variables
        train_x = train_data.iloc[:, :-1]
        train_y = train_data.iloc[:, -1]

        # Initial error
        error = 0

        # Go through each row of tune
        for row_index, row in tune_x.iterrows():
            # Calculate distances
            distances = ((train_x - row) ** 2).sum(axis=1).sort_values()[:k]

            # Grab neighbors
            neighbors = distances[:k]

            # Kernel calculation
            kernel = np.exp((1 / (2 * sigma)) * neighbors)

            # Grab actual values of neighbors
            neighbors_r = train_y.loc[neighbors.index.to_list()]

            # Calculate prediction
            prediction = sum(kernel * neighbors_r) / sum(kernel)

            # Grab the actual for comparison
            actual = tune_y.loc[row_index]

            # Add squared error
            error += (actual - prediction) ** 2

        # Average error to get MSE
        mse = error / len(tune_data)

        # Return MSE
        return mse

    def predict(self, k=None, sigma=None):
        """
        Predict function

        Function to predict on our test data set

        :param k: int, K, if non specified default to self.K, which is from the Tune function
        :param sigma: float, sigma, if non specified default to self.sigma, which is from the Tune function
        """
        # Default K
        if not k:
            k = self.k

        # Default Sigma
        if not sigma:
            sigma = self.sigma

        # Set up dictionary to hold end results
        test_results = {
            index: {
                'mse': 0,
                'prediction': []
            } for index in range(5)
        }

        # Loop through each of the test splits
        for index in range(5):
            # Define train
            train_data = self.train_data[index]
            train_x = train_data.iloc[:, :-1]
            train_y = train_data.iloc[:, -1]

            # Define test
            test_data = self.data_split[index]
            test_x = test_data.iloc[:, :-1]
            test_y = test_data.iloc[:, -1]

            # Loop through test data set
            for row_index, row in test_x.iterrows():
                # Calculate distances
                distances = ((train_x - row) ** 2).sum(axis=1).sort_values()

                # Define neighbors
                neighbors = distances[:k]

                # Kernel calculation
                kernel = np.exp((1 / (2 * sigma)) * neighbors)

                # Grab actual values of neighbors
                neighbors_r = train_y.loc[neighbors.index.to_list()]

                # Calculate prediction
                prediction = sum(kernel * neighbors_r) / sum(kernel)

                # Grab the actual for comparison
                actual = test_y.loc[row_index]

                # Add squared error
                test_results[index]['mse'] += (actual - prediction) ** 2

                # Append prediction to results
                test_results[index]['prediction'].append(prediction)

            # Average error to get MSE
            test_results[index]['mse'] = test_results[index]['mse'] / len(test_data)

        # Set test_results
        self.test_results = test_results

    def output(self):
        """
        Output function to output results

        :return JSON: summary of tune and train
        :return CSV: csv of classification over train data sets
        """
        # Get MSE
        mse = sum([self.test_results[index]['mse'] for index in range(5)])

        # Summary JSON
        self.summary = {
            'tune': {
                'k': self.k,
                'sigma': self.sigma
            },
            'test': {
                'mse': mse / 5
            }
        }

        # If edited or condensed also add epsilon
        if self.knn_type != 'regular':
            self.summary['tune'].update({'epsilon': self.epsilon})

        # Output JSON
        with open(f'output_{self.data_name}\\{self.data_name}_{self.knn_type}_summary.json', 'w') as file:
            json.dump(self.summary, file)

        # Summary CSV
        summary_prediction = pd.DataFrame()

        # Loop through each test data set and add the results
        for index in range(5):
            temp_summary_prediction = self.data_split[index]
            temp_summary_prediction['prediction'] = self.test_results[index]['prediction']

            # Append temp to the CSV
            summary_prediction = summary_prediction.append(temp_summary_prediction)

        # Dump CSV and save
        summary_prediction.to_csv(f'output_{self.data_name}\\{self.data_name}_{self.knn_type}_prediction.csv')
        self.summary_prediction = summary_prediction

    def visualize_tune(self):
        """
        Tune visualization function

        This function uses the results of the tune function to create a plot graph

        :return: matplotlib saved jpg in output folder
        """
        # Define parameters tunes
        tune_list = ['k', 'sigma']

        # If edited or condensed add epsilon
        if self.knn_type != 'regular':
            tune_list.append('epsilon')

        # For each parameter make a visualization
        for parameter in tune_list:
            # Grab optimal value
            if parameter == 'k':
                optimal = self.k
            elif parameter == 'sigma':
                optimal = self.sigma
            else:
                optimal = self.epsilon

            # Figure / axis set up
            fig, ax = plt.subplots()

            # We'll plot the list of params and their accuracy
            ax.plot(self.tune_results[parameter].keys(), self.tune_results[parameter].values())

            # Title
            ax.set_title(rf'{self.data_name} {parameter} Tune Results - Optimal: {parameter} {optimal}')

            # X axis
            ax.set_xlabel(parameter)
            ax.set_xticks(list(self.tune_results[parameter].keys()))
            ax.set_xticklabels(list(self.tune_results[parameter].keys()), rotation=45, fontsize=6)

            # Y axis
            ax.set_ylabel('MSE')

            # Saving
            if parameter == 'epsilon':
                plt.savefig(f'output_{self.data_name}\\{self.data_name}_{self.knn_type}_{parameter}_tune.jpg')
            else:
                plt.savefig(f'output_{self.data_name}\\{self.data_name}_{parameter}_tune.jpg')
