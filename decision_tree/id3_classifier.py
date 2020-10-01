import math


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

            normalizer = len(train_data)
            entropy = 0

            for class_name in self.class_names:
                class_count = len(train_data.loc[train_data['Class'] == class_name])
                entropy += - (class_count / normalizer) * math.log((class_count / normalizer), 2)

            for feature_name in self.feature_names.keys():
                expectation = 0

                if self.feature_names[feature_name] == 'categorical':
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
