import math
import pandas as pd
import json


class ID3Classifier:
    """
    Class ID3 Classifier

    The ID3 algorithm is a decision tree algorithm that performs splits using the information gain criteria. This class
        implements a fit, predict, and prune function. The prune function implements a post pruning process against
        the validation set by comparing misclassification error.
    """
    def __init__(self, etl, prune=False):
        """
        Init function

        Sets main variables and determines pruning

        :param etl: etl, etl object with transformed and split data
        :param prune: bool, boolean to set prune against fully grown train model
        """
        # Set the attributes to hold our data
        self.etl = etl
        self.data_name = self.etl.data_name
        self.validation_data = etl.validation_data
        self.test_split = etl.test_split
        self.train_split = etl.train_split
        self.class_names = etl.class_names
        self.feature_names = etl.feature_names
        self.prune_tree = prune

        # Train Models
        self.train_models = {}

        # Validation Results
        self.validation_results = {}

        # Test Results
        self.test_results = {}

        # Summary
        self.summary = {}
        self.summary_classification = None

    def fit(self):
        """
        Fit function

        Loops through each of the train splits (combined 4 CV splits) and fits a tree to each of them. If prune is
            specified that is also handled here.
        """
        # Loop through each train split
        for train_index in range(5):
            # Set data and feature names
            train_data = self.train_split[train_index]
            initial_features = list(self.feature_names.keys())

            # Branch
            tree = self.branch(train_data, initial_features)

            # Update models with the final tree
            self.train_models.update({train_index: tree})

        # If prune, perform pruning
        if self.prune_tree:
            self.prune()

    def branch(self, train_data, feature_names):
        """
        Branch function

        This function takes a data set, and checks for the best split among all of the features. For categorical
            features calculate_expectation_categorical is called. For numerical features calculate_expectation_numeric
            is called. The feature types are set during the ETL. This is a recursive function that calls to itself
            to continue to check for branches until the stop condition is met. The stop condition is:
                -pure data set (single class)
                
        :param train_data: DataFrame, data set to perform information calculation against
        :param feature_names: list, list of features to perform information calculation on
        :return tree: dictionary or DataFrame, if the DataFrame is pure (one class) a DataFrame is returned as a leaf
            otherwise a tree pointing at partitions and their sub trees is returned
        """
        # Initial variables
        normalizer = len(train_data)
        entropy = 0
        max_gain = 0
        max_feature_name = None
        max_partition = None

        # Calculation of entropy over the data set
        for class_name in self.class_names:
            # For each class, calculate an entropy
            class_count = len(train_data.loc[train_data['Class'] == class_name])
            if class_count != 0:
                entropy += - (class_count / normalizer) * math.log((class_count / normalizer), 2)

        # Stop condition, if entropy is 0, we are at a pure data set and need to return a leaf
        if entropy == 0:
            return train_data
        
        # Begin gain calculation for each feature
        for feature_name in feature_names:
            chosen_partition = None

            # Call to calculate function for categorical features
            if self.feature_names[feature_name] == 'categorical':
                expectation = self.calculate_expectation_categorical(train_data=train_data, feature_name=feature_name)

            # Call to calculate function for numerical features
            else:
                expectation, chosen_partition = self.calculate_expectation_numerical(train_data=train_data,
                                                                                     feature_name=feature_name)

            # Gain calculation
            gain = entropy - expectation

            # Check feature gain against current max gain
            if gain > max_gain:
                # If gain is larger, save new max gain, name, and the partition (if numerical)
                max_gain = gain
                max_feature_name = feature_name
                max_partition = chosen_partition

        # If there is a max partition, our feature is numerical
        if max_partition:
            # Split data into upper and lower around the chosen partition
            lower_new_train_data = train_data.loc[train_data[max_feature_name] <= max_partition]
            upper_new_train_data = train_data.loc[train_data[max_feature_name] > max_partition]

            # Double recursive call for the upper and lower data sets, set to tree for the chosen feature
            tree = {
                max_feature_name: {
                    f'<{max_partition}':
                        self.branch(train_data=lower_new_train_data, feature_names=feature_names),
                    f'>{max_partition}':
                        self.branch(train_data=upper_new_train_data, feature_names=feature_names)
                }
            }

        # Otherwise our feature is categorical
        else:
            # Grab the categorical partitions
            max_feature_partitions = train_data[max_feature_name].unique().tolist()
            
            # Empty tree initialization
            tree = {max_feature_name: {partition: {} for partition in max_feature_partitions}}

            # For each categorical partition, recursive call
            for partition in max_feature_partitions:
                # Filter for the current categorical partition
                new_train_data = train_data.loc[train_data[max_feature_name] == partition]
                
                # Retrieve branch for the current categorical partition
                next_branch = self.branch(train_data=new_train_data, feature_names=feature_names)

                # Update tree at the categorical partition with the returned tree
                tree[max_feature_name].update({partition: next_branch})

        # Return our tree
        return tree

    def calculate_expectation_categorical(self, train_data, feature_name):
        """
        Expectation function - categorical
        
        This function handles calculation of expectation for a categorical feature
        
        :param train_data: DataFrame, data set to perform information calculation against
        :param feature_name: str, name of feature to perform information calculation on
        :return expectation: float, expected gain from partitioning on this feature
        """
        # Initial variables
        normalizer = len(train_data)
        expectation = 0

        # Retrieve the categorical partitions
        partitions = train_data[feature_name].unique().tolist()

        # For each partition
        for partition in partitions:
            # Grab partition variables to calculate entropy
            partition_count = len(train_data.loc[train_data[feature_name] == partition])
            partition_entropy = 0

            # Add to entropy over each class in this partition
            for class_name in self.class_names:
                partition_class_count = len(train_data.loc[(train_data[feature_name] == partition) &
                                                           (train_data['Class'] == class_name)])
                if partition_class_count != 0:
                    partition_entropy += - (partition_class_count / partition_count) * \
                                         math.log((partition_class_count / partition_count), 2)

            # Sum up expectation over all the classes in this partition
            expectation += (partition_count / normalizer) * partition_entropy

        # Return final expectation
        return expectation

    def calculate_expectation_numerical(self, train_data, feature_name):
        """
        Expectation function - numerical

        This function handles calculation of expectation for a numerical feature. This function also tests the
            expectation over a few partitions around class splits and returns the best one

        :param train_data: DataFrame, data set to perform information calculation against
        :param feature_name: str, name of feature to perform information calculation on
        :return expectation: float, expected gain from partitioning on this feature
        :return chosen_partition: float, best partition for this feature
        """
        # Initial variables
        normalizer = len(train_data)
        expectation = 0
        chosen_partition = None

        # Choosing of partitions to test on
        partitions = []

        # For each class, calculate the mid point between the max and min of this class and non class
        for class_name in self.class_names:
            if len(train_data.loc[train_data['Class'] == class_name]) > 0:
                partitions.append((train_data.loc[train_data['Class'] == class_name][feature_name].max() +
                                   train_data.loc[train_data['Class'] != class_name][feature_name].min()) / 2)
                partitions.append((train_data.loc[train_data['Class'] == class_name][feature_name].min() +
                                   train_data.loc[train_data['Class'] != class_name][feature_name].max()) / 2)

        # Make sure we only check distinct partitions
        partitions = set(partitions)

        # Calculate expectation over each partition
        for partition in partitions:
            # Set initial variables over each partition
            partition_expectation = 0
            lower_partition_entropy = 0
            upper_partition_entropy = 0
            lower_partition_count = len(train_data.loc[train_data[feature_name] <= partition])
            upper_partition_count = len(train_data.loc[train_data[feature_name] > partition])

            # For each class calculate entropy over the upper and lower data sets of the partition
            for class_name in self.class_names:
                lower_partition_class_count = len(train_data.loc[(train_data[feature_name] <= partition) &
                                                                 (train_data['Class'] == class_name)])

                upper_partition_class_count = len(train_data.loc[(train_data[feature_name] > partition) &
                                                                 (train_data['Class'] == class_name)])

                if lower_partition_class_count != 0:
                    lower_partition_entropy += - (lower_partition_class_count / lower_partition_count) * \
                                               math.log((lower_partition_class_count / lower_partition_count),
                                                        2)

                if upper_partition_class_count != 0:
                    upper_partition_entropy += - (upper_partition_class_count / upper_partition_count) * \
                                               math.log((upper_partition_class_count / upper_partition_count),
                                                        2)

            # Add entropy for this partition
            partition_expectation += (lower_partition_count / normalizer) * lower_partition_entropy
            partition_expectation += (upper_partition_count / normalizer) * upper_partition_entropy

            # Expectation comparison, if this is currently smaller than previous, it is our new chosen
            if partition_expectation < expectation:
                chosen_partition = partition
                expectation = partition_expectation
            # Set initial expectation to the first partition
            elif expectation == 0:
                chosen_partition = partition
                expectation = partition_expectation

        # Return expectation and the chosen partition
        return expectation, chosen_partition

    def prune(self):
        """
        Prune function

        This function loops over the fully grown trees and prunes them based on miscalculation error. This function
            calls to the prune_branch function which checks leaves against their parent tree.
        """
        # Check the single validation data set against each of the train models
        for index in range(5):
            # Assign tree
            tree = self.train_models[index]

            # Retrieve results, and the edited model
            validation_result, train_model, new_leaf = self.prune_branch(prediction_data=self.validation_data,
                                                                         tree=tree)

            # Set edited model and results
            self.train_models.update({index: train_model})
            self.validation_results.update({index: validation_result})

    def prune_branch(self, prediction_data, tree):
        """
        Prune branch function

        This function compared against the passed tree and the data the misclassification error. The Tree is compared
            with a new leaf, the combined dataset of its child leaves. If the new leaf is determine to have lower
            misclassification error, the new leaf is returned to replace the tree. This is a recursive function with
            two stopping conditions:
                -leaf node: if the tree is a leaf, return it
                -validation_data is empty: if the validation data is empty, this branch has over fit, return the leaf
            Otherwise a tree is returned

        :param prediction_data: DataFrame, data set to check for pruning against tree
        :param tree: dictionary or DataFrame, current tree and sub tree to check (or leaf)
        :return validation_result: DataFrame, prediction_data with prediction class added
        :return tree: dictionary of DataFrame, if a stopping condition was met, a DataFrame is return, else a dictionary
            Whatever is returned will replace the original tree
        :return leaf: DataFrame, a combined dataframe of a tree and all of its subtrees
        """
        # Initial Variables
        validation_result = pd.DataFrame()
        new_leaf = pd.DataFrame()

        # First stop condition - if the tree is a DataFrame it is a leaf, perform prediction and return
        if isinstance(tree, pd.DataFrame):
            # Prediction is made based on the mode class
            prediction = tree['Class'].mode()[0]
            prediction_data['Prediction'] = prediction

            return prediction_data, tree, tree

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
                new_validation_result, branch, leaf = self.prune_branch(prediction_data=new_prediction_data,
                                                                        tree=new_tree)

                # Set the current partition of our tree to the new branch
                tree[feature_name][partition] = branch

                # Append results
                validation_result = validation_result.append(new_validation_result)

                # Append leaves together to form a new leaf
                new_leaf = new_leaf.append(leaf)

            # If we have no validation data, we have over fit and should replace the tree with the new leaf
            if len(validation_result) == 0:
                return validation_result, new_leaf, new_leaf

            # Perform prediction of class over the new leaf
            new_classification = new_leaf['Class'].mode()[0]
            new_misclassification = len(prediction_data.loc[prediction_data['Class'] != new_classification]) / \
                                    len(prediction_data)

            # If any validation results were missed from filtering, we need to readd and classify
            missed_validation_results = \
                pd.DataFrame.copy(prediction_data.loc[~prediction_data.index.isin(validation_result.index)], deep=True)
            if len(missed_validation_results) > 0:
                missed_validation_results['Prediction'] = new_classification

                validation_result = validation_result.append(missed_validation_results)

            # Calculate old misclassification where our results are from the subtrees
            old_misclassification = len(validation_result.loc[validation_result['Class'] !=
                                                              validation_result['Prediction']]) /\
                                    len(validation_result)

            # If the new leaf is better than the old tree, replace the tree with the new leaf
            if new_misclassification <= old_misclassification:
                validation_result['Prediction'] = new_classification

                return validation_result, new_leaf, new_leaf

            # Otherwise, return the tree with any changed sub trees
            else:
                return validation_result, tree, new_leaf

    def predict(self):
        """
        Predict Function

        This function loops through each of the 5 CV splits and classifies against the train models. The train models
            may be full or pruned depending on how it was trained
        """
        # Loop through each CV split
        for index in range(5):
            # Define data set and corresponding model
            test_data = self.test_split[index]
            tree = self.train_models[index]

            # Classify
            test_result, leaf = self.classify(prediction_data=test_data, tree=tree)

            # Save results
            self.test_results.update({index: test_result})

    def classify(self, prediction_data, tree):
        """
        Classify function

        This function is called by the predict function. It makes a recursive call by traveling down the tree, filtering
            data as it does. There is one stop condition:
                -leaf node: if a tree is a leaf node, make prediction and return

        :param prediction_data: DataFrame, data set to check for pruning against tree
        :param tree: dictionary or DataFrame, current tree and sub tree to check (or leaf)
        :return validation_result: DataFrame, prediction_data with prediction class added
        :return leaf: DataFrame, a combined dataframe of a tree and all of its subtrees, used in the case if all the
            branches miss a data point during filtering
        """
        # Initial variables
        test_result = pd.DataFrame()
        new_leaf = pd.DataFrame()

        # First stop condition - if the tree is a DataFrame it is a leaf, perform prediction and return
        if isinstance(tree, pd.DataFrame):
            # Prediction is made based on the mode class
            prediction = tree['Class'].mode()[0]
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
                new_prediction_data, leaf = self.classify(prediction_data=new_prediction_data, tree=new_tree)

                # Append results
                test_result = test_result.append(new_prediction_data)

                # Append leaves together to form a new leaf
                new_leaf = new_leaf.append(leaf)

        # If all of the subtrees miss a datapoint, we need to assign a value to it, use new_leaf for that
        new_classification = new_leaf['Class'].mode()[0]

        # If any validation results were missed from filtering, we need to readd and classify
        missed_validation_results = \
            pd.DataFrame.copy(prediction_data.loc[~prediction_data.index.isin(test_result.index)], deep=True)
        if len(missed_validation_results) > 0:
            missed_validation_results['Prediction'] = new_classification

            test_result = test_result.append(missed_validation_results)

        # Return result and the created new leaf
        return test_result, new_leaf

    def filter_data(self, prediction_data, feature_name, partition):
        """
        Helper function to filter data

        This data takes data and a filter name, and attempts to filter over all the partitions. This function is used
            by the classify and prune functions.

        :param prediction_data: DataFrame, data set to check for pruning against tree
        :param feature_name: str, name of feature to filter
        :param partition: str, partition to filter on
        :return prediction_data: DataFrame, filtered on partition and feature
        """
        if not self:
            raise NotImplementedError

        # For the numerical features, a less and greater than was appended to the beginning to define upper and lower
        if partition[0] == '<':
            # Lower, split off sign and cast to float
            float_partition = float(partition[1:])

            return pd.DataFrame.copy(prediction_data.loc[prediction_data[feature_name] <= float_partition], deep=True)

        # Numerical upper
        elif partition[0] == '>':
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
        # To change the file name of pruned trees
        if self.prune_tree:
            prune_file_name = 'pruned_'
        else:
            prune_file_name = ''

        # Initial Variables
        all_test_results = pd.DataFrame()
        misclassification = 0

        # Combine the test results into one single data set and average out the misclassification error
        for index in range(5):
            # Append results
            test_results = self.test_results[index]
            all_test_results = all_test_results.append(test_results)

            # Add misclassification for this index
            misclassification += len(test_results.loc[test_results['Class'] != test_results['Prediction']]) /\
                                 len(test_results)

        # CSV output
        all_test_results.to_csv(f'output_{self.data_name}\\{self.data_name}_{prune_file_name}test_results.csv')

        # Calculate average misclassification
        misclassification = misclassification / 5

        # Save
        self.summary = {
            'test': {
                'misclassification': misclassification
            }
        }

        # Output JSON
        with open(f'output_{self.data_name}\\{self.data_name}_{prune_file_name}summary.json', 'w') as file:
            json.dump(self.summary, file)
