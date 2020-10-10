import copy
import pandas as pd
import json


class CARTRegressor:
    """
    Class CART Regressor

    The CART algorithm is a decision tree algorithm that splits using MSE. It implements a fit, predict, and tune
        function. The tune function uses a percentage of average squared value as a threshold. If a split has a MSE
        lower than the threshold, the algorithm stops splitting
    """
    def __init__(self, etl, percent_threshold=0):
        """
        Init function

        Sets main variables and determine tuning

        :param etl: etl, etl object with transformed and split data
        :param percent_threshold: float, percentage of squared value
        """
        # Set the attributes to hold our data
        self.etl = etl
        self.data_name = self.etl.data_name
        self.validation_data = etl.validation_data
        self.test_split = etl.test_split
        self.train_split = etl.train_split
        self.feature_names = etl.feature_names
        self.squared_average_target = etl.squared_average_target

        # Tune Results
        self.percent_threshold = percent_threshold
        self.tune_results = {}
        self.perform_tune = False

        # Train Models
        self.train_models = {}

        # Test Results
        self.test_results = {}

        # Summary
        self.summary = {}
        self.tune_summary = None

    def tune(self):
        """
        Tune function

        This function tests a few percent thresholds. The threshold is multiplied against the squared value of the
            target variable. The early stopped models are tested against a validation set. The MSE is then returned for
            each threshold
        """
        # Initial Variables
        self.perform_tune = True
        percent_threshold_list = [0, .01, .05, .1, .5]
        self.tune_results = {percent_threshold: {} for percent_threshold in percent_threshold_list}

        # Loop through each threshold
        for percent_threshold in percent_threshold_list:
            # Fit using threshold
            self.fit(percent_threshold)

            # Build a model over each of the train splits
            for index in range(5):
                # Build tree
                tune_results, tree = self.regress(self.validation_data, self.train_models[index])

                # Calculate MSE
                mse = ((tune_results.iloc[:, -2] - tune_results.iloc[:, -1]) ** 2).sum()
                mse = mse / len(tune_results)

                # Store results
                self.tune_results[percent_threshold].update({
                    index: mse
                })

    def fit(self, percent_threshold=None):
        """
        Fit function

        This function loops through each of the train splits (combined 4 CV splits) and fits a tree to each of them. For
            early stopping, a percent threshold can be fed to this function
        """
        # Retrieve percent threshold and calculate
        if not percent_threshold:
            percent_threshold = self.percent_threshold
        threshold = percent_threshold * self.squared_average_target

        # Loop through each train split
        for train_index in range(5):
            # Set data and feature names
            train_data = self.train_split[train_index]
            initial_features = list(self.feature_names.keys())

            # Branch
            tree = self.branch(train_data, initial_features, threshold)

            # Update models with the final tree
            self.train_models.update({train_index: tree})

    def branch(self, train_data, feature_names, threshold):
        """
        Branch function

        This function takes a data set, and checks for the best split among all of the features. For categorical
            features calculate_mse_categorical is called. For numerical features calculate_mse_numeric is called. The
            feature types are set during the ETL. This is a recursive function that calls to itself to continue to check
            for branches until the stop condition is met. The stop conditions are:
                -empty data frame (full tree)
                -MSE below threshold (stopped tree)

        :param train_data: DataFrame, data set to perform information calculation against
        :param feature_names: list, list of features to perform information calculation on
        :param threshold: float, threshold to stop splitting. If 0 the split only ends with no data
        :return tree: dictionary or DataFrame, if the DataFrame is pure (one class) a DataFrame is returned as a leaf
            otherwise a tree pointing at partitions and their sub trees is returned
        """
        # First stop condition: fully split data set
        if len(train_data) == 0:
            return train_data

        # Initial variables
        feature_names = copy.deepcopy(feature_names)
        min_mse = 0
        min_feature_name = None
        min_partition = None

        # Calculate an initial MSE. This is set to the current min
        partition_prediction = train_data.iloc[:, -1].mean()
        min_mse += ((train_data.iloc[:, -1] - partition_prediction) ** 2).sum()
        min_mse = min_mse / len(train_data)

        # If the current min is below threshold, stop and return the leaf
        if min_mse < threshold:
            return train_data

        # Loop over the features
        for feature_name in feature_names:
            chosen_partition = None

            # Call to calculate function for categorical features
            if self.feature_names[feature_name] == 'categorical':
                feature_mse = self.calculate_mse_categorical(train_data=train_data, feature_name=feature_name)

            # Call to calculate function for numerical features
            else:
                feature_mse, chosen_partition = self.calculate_mse_numerical(train_data=train_data,
                                                                             feature_name=feature_name)

            # Compare MSE, if lower than current update variables
            if feature_mse < min_mse:
                min_mse = feature_mse
                min_feature_name = feature_name
                min_partition = chosen_partition

        # If a min feature name was set, we have a feature to branch on
        if min_feature_name:
            # If there was a partition set, trigger numerical branching
            if min_partition:
                # Split data into upper and lower around the chosen partition
                lower_new_train_data = train_data.loc[train_data[min_feature_name] <= min_partition]
                upper_new_train_data = train_data.loc[train_data[min_feature_name] > min_partition]

                # Double recursive call for the upper and lower data sets, set to tree for the chosen feature
                tree = {
                    min_feature_name: {
                        f'<{min_partition}':
                            self.branch(train_data=lower_new_train_data, feature_names=feature_names,
                                        threshold=threshold),
                        f'>{min_partition}':
                            self.branch(train_data=upper_new_train_data, feature_names=feature_names,
                                        threshold=threshold)
                    }
                }

            # Otherwise our feature is categorical
            else:
                # Grab the categorical partitions
                min_feature_partitions = train_data[min_feature_name].unique().tolist()

                # Empty tree initialization
                tree = {min_feature_name: {partition: {} for partition in min_feature_partitions}}

                # For each categorical partition, recursive call
                for partition in min_feature_partitions:
                    # Filter for the current categorical partition
                    new_train_data = train_data.loc[train_data[min_feature_name] == partition]

                    # Retrieve branch for the current categorical partition
                    next_branch = self.branch(train_data=new_train_data, feature_names=feature_names,
                                              threshold=threshold)

                    # Update tree at the categorical partition with the returned tree
                    tree[min_feature_name].update({partition: next_branch})

            # Return our tree
            return tree

        # If our current leaf was better than its sub trees, stop branching and return the leaf
        else:
            return train_data

    def calculate_mse_categorical(self, train_data, feature_name):
        """
        MSE function - categorical

        This function handles calculation of MSE for a categorical feature

        :param train_data: DataFrame, data set to perform MSE calculation against
        :param feature_name: str, name of feature to perform MSE calculation on
        :return MSE: float, MSE from partitioning on this feature
        """
        if not self:
            raise NotImplementedError

        # Initial variables
        mse = 0

        # Retrieve the categorical partitions
        partitions = train_data[feature_name].unique().tolist()

        # For each partition
        for partition in partitions:
            # Grab partition split to calculate MSE
            partition_data = train_data.loc[train_data[feature_name] == partition]

            # Make a prediction
            partition_prediction = partition_data.iloc[:, -1].mean()

            # Add MSE
            mse += ((partition_data.iloc[:, -1] - partition_prediction) ** 2).sum()

        # Average MSE and return
        return mse / len(train_data)

    def calculate_mse_numerical(self, train_data, feature_name):
        """
        MSE function - numerical

        This function handles calculation of MSE for a numerical feature. This function also tests the MSE over a few
            partitions around class splits and returns the best one

        :param train_data: DataFrame, data set to perform MSE calculation against
        :param feature_name: str, name of feature to perform MSE calculation on
        :return expectation: float, MSE from partitioning on this feature
        :return chosen_partition: float, best partition for this feature
        """
        if not self:
            raise NotImplementedError

        # Initial variables
        mse = 0
        chosen_partition = None

        # Choosing of partitions to test on
        partitions = []
        for quantile in [.4, .45, .5, .55, .6]:
            partitions.append(train_data[feature_name].quantile(quantile))

        # Make sure we only check distinct partitions
        partitions = set(partitions)

        # Calculate MSE over each partition
        for partition in partitions:
            partition_mse = 0

            # For each class calculate MSE over the upper and lower data sets of the partition
            lower_partition_data = train_data.loc[train_data[feature_name] <= partition]
            lower_partition_prediction = lower_partition_data.iloc[:, -1].mean()
            partition_mse += ((lower_partition_data.iloc[:, -1] - lower_partition_prediction) ** 2).sum()

            upper_partition_data = train_data.loc[train_data[feature_name] > partition]
            upper_partition_prediction = upper_partition_data.iloc[:, -1].mean()
            partition_mse += ((upper_partition_data.iloc[:, -1] - upper_partition_prediction) ** 2).sum()

            # Average MSE for this partition
            partition_mse = partition_mse / len(train_data)

            # Comparison of partitions, if this one is the best, update values
            if partition_mse < mse:
                chosen_partition = partition
                mse = partition_mse
            elif mse == 0:
                chosen_partition = partition
                mse = partition_mse

        # Return best results
        return mse, chosen_partition

    def predict(self):
        """
        Predict Function

        This function loops through each of the 5 CV splits and predicts against the train models. The train models
            may be full or stopped depending on how it was trained
        """
        # Loop through each CV split
        for index in range(5):
            # Define data set and corresponding model
            test_data = self.test_split[index]
            tree = self.train_models[index]

            # Regress
            test_result, leaf = self.regress(prediction_data=test_data, tree=tree)

            # Save results
            self.test_results.update({index: test_result})

    def regress(self, prediction_data, tree):
        """
        Regress function

        This function is called by the predict function. It makes a recursive call by traveling down the tree, filtering
            data as it does. There is one stop condition:
                -leaf node: if a tree is a leaf node, make prediction and return

        :param prediction_data: DataFrame, data set to check for pruning against tree
        :param tree: dictionary or DataFrame, current tree and sub tree to check (or leaf)
        :return validation_result: DataFrame, prediction_data with prediction class added
        :return leaf: DataFrame, a combined dataframe of a tree and all of its subtrees, used in the case if all the
            branches miss a data point during filtering
        """
        # Initial Variables
        test_result = pd.DataFrame()
        new_leaf = pd.DataFrame()

        # First stop condition - if the tree is a DataFrame it is a leaf, perform prediction and return
        if isinstance(tree, pd.DataFrame):
            # Prediction is made based on the mean
            prediction = tree.iloc[:, -1].mean()
            prediction_data['Prediction'] = prediction

            return prediction_data, tree

        # Loop through the tree and its partition
        for feature_name in tree.keys():
            # Check each partition
            for partition in tree[feature_name]:
                # Set a variable for the sub tree to recursively call to
                new_tree = tree[feature_name][partition]

                # Filter based on the current partition, call to filter_data
                kwargs = {
                    'prediction_data': prediction_data,
                    'feature_name': feature_name,
                    'partition': partition
                }
                new_prediction_data = self.filter_data(**kwargs)

                # Recursive call now that we have the sub tree and filtered data
                new_prediction_data, leaf = self.regress(prediction_data=new_prediction_data, tree=new_tree)

                # Append results
                test_result = test_result.append(new_prediction_data)

                # Append leaves together to form a new leaf
                new_leaf = new_leaf.append(leaf)

            # If all of the subtrees miss a datapoint, we need to assign a value to it, use new_leaf for that
            missed_validation_results = prediction_data.loc[~prediction_data.index.isin(test_result.index)]

            # If any validation results were missed from filtering, we need to readd and regress
            if len(missed_validation_results) > 0:
                prediction = new_leaf.iloc[:, -1].mean()
                missed_validation_results['Prediction'] = prediction

                test_result = test_result.append(missed_validation_results)

        # Return result and the created new lea
        return test_result, new_leaf

    def filter_data(self, prediction_data, feature_name, partition):
        """
        Helper function to filter data

        This data takes data and a filter name, and attempts to filter over all the partitions. This function is used
            by the regress function.

        :param prediction_data: DataFrame, data set to check for pruning against tree
        :param feature_name: str, name of feature to filter
        :param partition: str, partition to filter on
        :return prediction_data: DataFrame, filtered on partition and feature
        """
        if not self:
            raise NotImplementedError

        # For the numerical features, a less and greater than was appended to the beginning to define upper and lower
        if str(partition)[0] == '<':
            # Lower, split off sign and cast to float
            float_partition = float(partition[1:])

            return pd.DataFrame.copy(prediction_data.loc[prediction_data[feature_name] <= float_partition], deep=True)

        # Numerical upper
        elif str(partition)[0] == '>':
            # Upper, split off sign and cast to float
            float_partition = float(partition[1:])

            return pd.DataFrame.copy(prediction_data.loc[prediction_data[feature_name] > float_partition], deep=True)

        # For Categorical variables, a simple filter is done on feature name and partition
        else:
            return pd.DataFrame.copy(prediction_data.loc[prediction_data[feature_name] == partition], deep=True)

    def summarize(self):
        """
        Summarize function

        Create summary JSON and CSV outputs to put into output folder
        """
        # To change the file name of stopped trees
        if self.percent_threshold > 0:
            stopped_file_name = 'stopped_'
        else:
            stopped_file_name = 'full_'

        # Initial Variables
        all_test_results = pd.DataFrame()
        average_mse = 0
        mse = 0

        # Combine the test results into one single data set and average out the MSE
        for index in range(5):
            # Append results
            test_results = self.test_results[index]
            all_test_results = all_test_results.append(test_results)

            # Add MSE for this index
            mse += ((test_results.iloc[:, -2] - test_results.iloc[:, -1]) ** 2).sum()
            mse = mse / len(test_results)

            # Average for this index
            average_mse += mse

        # CSV output
        all_test_results.to_csv(f'output_{self.data_name}\\{self.data_name}_{stopped_file_name}test_results.csv')

        # Calculate average MSE
        average_mse = average_mse / 5

        # Save
        self.summary = {
            'test': {
                'threshold': self.percent_threshold,
                'mse': average_mse
            }
        }

        # Output JSON
        with open(f'output_{self.data_name}\\{self.data_name}_{stopped_file_name}summary.json', 'w') as file:
            json.dump(self.summary, file)

        # If a tune was performed we'll also average over each of the thresholds
        if self.perform_tune:
            # Initial Variables
            self.tune_summary = {}

            # Loop through each percent threshold
            for percent_threshold in self.tune_results.keys():
                mse = 0

                # Add up MSE
                for index in range(5):
                    mse += self.tune_results[percent_threshold][index]

                # Average
                mse = mse / 5

                # Save
                self.tune_summary.update({percent_threshold: mse})

            # Output JSON
            with open(f'output_{self.data_name}\\{self.data_name}_tune_summary.json', 'w') as file:
                json.dump(self.tune_summary, file)
