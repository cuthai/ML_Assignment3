import copy
import pandas as pd


class CARTRegressor:
    def __init__(self, etl, percent_threshold=0):
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

        # Train Results
        self.train_models = {}

        # Test Results
        self.test_results = {}

        # Summary
        self.summary = {}

    def fit(self, threshold=None):
        if not threshold:
            threshold = self.percent_threshold * self.squared_average_target

        for train_index in range(5):
            train_data = self.train_split[train_index]
            initial_features = list(self.feature_names.keys())
            tree = self.branch(train_data, initial_features, threshold)

            self.train_models.update({train_index: tree})

    def branch(self, train_data, feature_names, threshold):
        feature_names = copy.deepcopy(feature_names)

        if len(train_data) == 0:
            return train_data

        min_mse = 0
        min_feature_name = None
        min_partition = None

        partition_prediction = train_data.iloc[:, -1].mean()
        min_mse += ((train_data.iloc[:, -1] - partition_prediction) ** 2).sum()
        min_mse = min_mse / len(train_data)

        if min_mse < threshold:
            return train_data

        for feature_name in feature_names:
            chosen_partition = None

            if self.feature_names[feature_name] == 'categorical':
                feature_mse = self.calculate_mse_categorical(train_data=train_data, feature_name=feature_name)

            else:
                feature_mse, chosen_partition = self.calculate_mse_numerical(train_data=train_data,
                                                                             feature_name=feature_name)

            if feature_mse < min_mse:
                min_mse = feature_mse
                min_feature_name = feature_name
                min_partition = chosen_partition

        if min_feature_name:
            if min_partition:
                lower_new_train_data = train_data.loc[train_data[min_feature_name] <= min_partition]
                upper_new_train_data = train_data.loc[train_data[min_feature_name] > min_partition]

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

            else:
                min_feature_partitions = train_data[min_feature_name].unique().tolist()
                tree = {min_feature_name: {partition: {} for partition in min_feature_partitions}}

                for partition in min_feature_partitions:
                    new_train_data = train_data.loc[train_data[min_feature_name] == partition]
                    next_branch = self.branch(train_data=new_train_data, feature_names=feature_names,
                                              threshold=threshold)

                    tree[min_feature_name].update({partition: next_branch})

            return tree

        else:
            return train_data

    def calculate_mse_categorical(self, train_data, feature_name):
        if not self:
            raise NotImplementedError

        mse = 0

        partitions = train_data[feature_name].unique().tolist()
        for partition in partitions:
            partition_data = train_data.loc[train_data[feature_name] == partition]
            partition_prediction = partition_data.iloc[:, -1].mean()
            mse += ((partition_data.iloc[:, -1] - partition_prediction) ** 2).sum()

        return mse / len(train_data)

    def calculate_mse_numerical(self, train_data, feature_name):
        if not self:
            raise NotImplementedError

        mse = 0
        chosen_partition = None

        partitions = []
        for quantile in [.4, .45, .5, .55, .6]:
            partitions.append(train_data[feature_name].quantile(quantile))

        partitions = set(partitions)

        for partition in partitions:
            partition_mse = 0

            lower_partition_data = train_data.loc[train_data[feature_name] <= partition]
            lower_partition_prediction = lower_partition_data.iloc[:, -1].mean()
            partition_mse += ((lower_partition_data.iloc[:, -1] - lower_partition_prediction) ** 2).sum()

            upper_partition_data = train_data.loc[train_data[feature_name] > partition]
            upper_partition_prediction = upper_partition_data.iloc[:, -1].mean()
            partition_mse += ((upper_partition_data.iloc[:, -1] - upper_partition_prediction) ** 2).sum()

            partition_mse = partition_mse / len(train_data)

            if partition_mse < mse:
                chosen_partition = partition
                mse = partition_mse
            elif mse == 0:
                chosen_partition = partition
                mse = partition_mse

        return mse, chosen_partition

    def predict(self):
        for index in range(5):
            test_data = self.test_split[index]
            tree = self.train_models[index]
            test_result, leaf = self.regress(prediction_data=test_data, tree=tree)

            self.test_results.update({index: test_result})

    def regress(self, prediction_data, tree):
        test_result = pd.DataFrame()
        new_leaf = pd.DataFrame()

        if isinstance(tree, pd.DataFrame):
            prediction = tree.iloc[:, -1].mean()
            prediction_data['Prediction'] = prediction

            return prediction_data, tree

        for feature_name in tree.keys():
            for partition in tree[feature_name]:
                new_tree = tree[feature_name][partition]

                kwargs = {
                    'prediction_data': prediction_data,
                    'feature_name': feature_name,
                    'partition': partition
                }
                new_prediction_data = self.filter_data(**kwargs)

                new_prediction_data, leaf = self.regress(prediction_data=new_prediction_data, tree=new_tree)

                test_result = test_result.append(new_prediction_data)
                new_leaf = new_leaf.append(leaf)

            missed_validation_results = prediction_data.loc[~prediction_data.index.isin(test_result.index)]

            if len(missed_validation_results) > 0:
                prediction = new_leaf.iloc[:, -1].mean()
                missed_validation_results['Prediction'] = prediction

                test_result = test_result.append(missed_validation_results)

        return test_result, new_leaf

    def filter_data(self, prediction_data, feature_name, partition):
        if not self:
            raise NotImplementedError

        if str(partition)[0] == '<':
            float_partition = float(partition[1:])

            return pd.DataFrame.copy(prediction_data.loc[prediction_data[feature_name] <= float_partition], deep=True)

        elif str(partition)[0] == '>':
            float_partition = float(partition[1:])

            return pd.DataFrame.copy(prediction_data.loc[prediction_data[feature_name] > float_partition], deep=True)

        else:
            return pd.DataFrame.copy(prediction_data.loc[prediction_data[feature_name] == partition], deep=True)

    def summarize(self):
        average_mse = 0
        mse = 0

        for index in range(5):
            test_results = self.test_results[index]

            mse += ((test_results.iloc[:, -2] - test_results.iloc[:, -1]) ** 2).sum()
            mse = mse / len(test_results)

            average_mse += mse

        average_mse = average_mse / 5

        self.summary = {
            'test': {
                'mse': average_mse
            }
        }
