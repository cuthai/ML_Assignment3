class ID3Classifier:
    def __init__(self, etl):
        # Set the attributes to hold our data
        self.etl = etl
        self.data_name = self.etl.data_name
        self.tune_data = etl.tune_data
        self.test_split = etl.test_split
        self.train_split = etl.train_split

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

