import math
import copy
import numpy as np


class ID3Classifier:
    def __init__(self, etl):
        # Set the attributes to hold our data
        self.etl = etl
        self.data_name = self.etl.data_name
        self.tune_data = etl.tune_data
        self.test_split = etl.test_split
        self.train_split = etl.train_split
        self.class_names = etl.class_names
        self.feature_names = etl.feature_names

        # Tune Results
        self.tune_results = {}
        self.k = 1

        # Test Results
        self.test_results = {}

        # Summary
        self.summary = {}
        self.summary_classification = None

    def fit(self):
        for train_index in range(5):
            train_data = self.train_split[train_index]
            initial_features = list(self.feature_names.keys())
            tree = self.branch(train_data, initial_features)

            pass

    def branch(self, train_data, feature_names):
        feature_names = copy.deepcopy(feature_names)

        normalizer = len(train_data)
        entropy = 0

        if normalizer == 0 or len(feature_names) == 0:
            return train_data

        for class_name in self.class_names:
            class_count = len(train_data.loc[train_data['Class'] == class_name])
            if class_count != 0:
                entropy += - (class_count / normalizer) * math.log((class_count / normalizer), 2)

        if entropy == 0:
            return train_data

        max_gain = 0
        max_feature_name = None
        max_partition = None

        for feature_name in feature_names:
            expectation = 0
            chosen_partition = None

            if self.feature_names[feature_name] == 'categorical':
                expectation = self.calculate_expectation_categorical(train_data=train_data, feature_name=feature_name)

            else:
                partitions = []
                for class_name in self.class_names:
                    if len(train_data.loc[train_data['Class'] == class_name]) > 0:
                        partitions.append((train_data.loc[train_data['Class'] == class_name][feature_name].max() +
                                           train_data.loc[train_data['Class'] != class_name][feature_name].min()) / 2)
                        partitions.append((train_data.loc[train_data['Class'] == class_name][feature_name].min() +
                                           train_data.loc[train_data['Class'] != class_name][feature_name].max()) / 2)

                partitions = set(partitions)

                for partition in partitions:
                    partition_expectation = 0
                    lower_partition_count = len(train_data.loc[train_data[feature_name] <= partition])
                    upper_partition_count = len(train_data.loc[train_data[feature_name] > partition])
                    lower_partition_entropy = 0
                    upper_partition_entropy = 0

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

                    partition_expectation += (lower_partition_count / normalizer) * lower_partition_entropy
                    partition_expectation += (upper_partition_count / normalizer) * upper_partition_entropy

                    if partition_expectation < expectation:
                        chosen_partition = partition
                        expectation = partition_expectation
                    elif expectation == 0:
                        chosen_partition = partition
                        expectation = partition_expectation

            gain = entropy - expectation

            if gain > max_gain:
                max_gain = gain
                max_feature_name = feature_name
                max_partition = chosen_partition

        if len(feature_names) > 0:
            if max_partition:
                lower_new_train_data = train_data.loc[train_data[max_feature_name] <= max_partition]
                upper_new_train_data = train_data.loc[train_data[max_feature_name] > max_partition]

                tree = {
                    max_feature_name: {
                        f'<{max_partition}':
                            self.branch(train_data=lower_new_train_data, feature_names=feature_names),
                        f'>{max_partition}':
                            self.branch(train_data=upper_new_train_data, feature_names=feature_names)
                    }
                }

            else:
                max_feature_partitions = train_data[max_feature_name].unique().tolist()
                tree = {max_feature_name: {partition: {} for partition in max_feature_partitions}}

                for partition in max_feature_partitions:
                    new_train_data = train_data.loc[train_data[max_feature_name] == partition]
                    next_branch = self.branch(train_data=new_train_data, feature_names=feature_names)

                    tree[max_feature_name].update({partition: next_branch})

            return tree

        else:
            return train_data

    def calculate_expectation_categorical(self, train_data, feature_name):
        normalizer = len(train_data)
        expectation = 0

        partitions = train_data[feature_name].unique().tolist()
        for partition in partitions:
            partition_count = len(train_data.loc[train_data[feature_name] == partition])
            partition_entropy = 0

            for class_name in self.class_names:
                partition_class_count = len(train_data.loc[(train_data[feature_name] == partition) &
                                                           (train_data['Class'] == class_name)])
                if partition_class_count != 0:
                    partition_entropy += - (partition_class_count / partition_count) * \
                                         math.log((partition_class_count / partition_count), 2)

            expectation += (partition_count / normalizer) * partition_entropy

        return expectation
