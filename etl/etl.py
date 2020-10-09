import pandas as pd
import numpy as np


class ETL:
    """
    Class ETL to handle the ETL of the data.

    This class really only does the extract and transform functions of ETL. The data is then received downstream by the
        algorithms for processing.
    """
    def __init__(self, data_name, random_state=1):
        """
        Init function. Takes a data_name and extracts the data and then transforms.

        All data comes from the data folder. The init function calls to both extract and transform for processing

        :param data_name: str, name of the data file passed at the command line. Below are the valid names:
            breast-cancer
            car
            segmentation
            abalone
            machine (assignment name: computer hardware)
            forest-fires
        :param random_state: int, seed for data split
        """
        # Set the attributes to hold our data
        self.data = None
        self.transformed_data = None
        self.validation_data = None
        self.test_split = {}
        self.train_split = {}

        # Meta attributes
        self.data_name = data_name
        self.random_state = random_state
        self.classes = 0
        self.class_names = None
        self.feature_names = None
        self.squared_average_target = 0

        # Extract
        self.extract()

        # Transform
        self.transform()

        # Split
        if self.classes == 0:
            self.cv_split_regression()
        else:
            self.cv_split_classification()

        # Combine Train Sets
        self.cv_combine()

    def extract(self):
        """
        Function to extract data based on data_name passed

        :return self.data: DataFrame, untransformed data set
        """
        # breast-cancer
        if self.data_name == 'breast-cancer':
            column_names = ['ID', 'Clump_Thickness', 'Uniformity_Cell_Size', 'Uniformity_Cell_Shape',
                            'Marginal_Adhesion', 'Single_Epithelial_Cell_Size', 'Bare_Nuclei', 'Bland_Chromatin',
                            'Normal_Nucleoli', 'Mitoses', 'Class']
            self.data = pd.read_csv('data\\breast-cancer-wisconsin.data', names=column_names)

        # car
        elif self.data_name == 'car':
            column_names = ['Buying', 'Maintenance', 'Doors', 'Persons', 'Luggage_Boot', 'Safety', 'Class']
            self.data = pd.read_csv('data\\car.data', names=column_names)

        # segmentation
        elif self.data_name == 'segmentation':
            column_names = ['Class', 'Region_Centroid_Col', 'Region_Centroid_Row', 'Region_Pixel_Count',
                            'Short_Line_Density_5', 'Short_Line_Density_2', 'Vedge_Mean', 'Vedge_SD', 'Hedge_Mean',
                            'Hedge_SD', 'Intensity_Mean', 'Raw_Red_Mean', 'Raw_Blue_Mean', 'Raw_Green_Mean',
                            'Ex_Red_Mean', 'Ex_Blue_Mean', 'Ex_Green_Mean', 'Value_Mean', 'Saturation_Mean', 'Hue_Mean']
            self.data = pd.read_csv('data\\segmentation.data', names=column_names, skiprows=5)

        # abalone
        elif self.data_name == 'abalone':
            column_names = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_Weight', 'Shucked_Weight', 'Viscera_Weight',
                            'Shell_Weight', 'Rings']
            self.data = pd.read_csv('data\\abalone.data', names=column_names)

        # machine
        elif self.data_name == 'machine':
            column_names = ['Vendor', 'Model_Name', 'MYCT', 'MMIN', 'MMAX', 'CACH', 'CHMIN', 'CHMAX', 'PRP', 'ERP']
            self.data = pd.read_csv('data\\machine.data', names=column_names)

        # forest-fires
        elif self.data_name == 'forest-fires':
            self.data = pd.read_csv('data\\forestfires.data')

        # If an incorrect data_name was specified we'll raise an error here
        else:
            raise NameError('Please specify a predefined name for one of the 6 data sets (breast-cancer, car,'
                            'segmentation, abalone, machine, forest-fires)')

    def transform(self):
        """
        Function to transform the specified data

        This is a manager function that calls to the actual helper transform function.
        """
        # breast-cancer
        if self.data_name == 'breast-cancer':
            self.transform_breast_cancer()

        # car
        elif self.data_name == 'car':
            self.transform_car()

        # segmentation
        elif self.data_name == 'segmentation':
            self.transform_segmentation()

        # abalone
        elif self.data_name == 'abalone':
            self.transform_abalone()

        # machine
        elif self.data_name == 'machine':
            self.transform_machine()

        # forest-fires
        elif self.data_name == 'forest-fires':
            self.transform_forest_fires()

        # The extract function should catch this but lets throw again in case
        else:
            raise NameError('Please specify a predefined name for one of the 6 data sets (glass, segmentation, vote,'
                            'abalone, machine, forest-fires)')

    def transform_breast_cancer(self):
        """
        Function to transform breast-cancer data set

        For this function missing data points are removed

        :return self.transformed_data: DataFrame, transformed data set
        :return self.classes: int, num of classes
        """
        # Remove missing data points
        self.data = self.data.loc[self.data['Bare_Nuclei'] != '?']
        self.data['Bare_Nuclei'] = self.data['Bare_Nuclei'].astype(int)
        self.data.reset_index(inplace=True, drop=True)

        # We'll make a deep copy of our data set
        temp_df = pd.DataFrame.copy(self.data, deep=True)

        # We don't need ID so let's drop that
        temp_df.drop(columns='ID', inplace=True)

        # Set attributes for ETL object
        self.classes = 2
        self.transformed_data = temp_df

        # Class and Feature name/type
        self.class_names = temp_df['Class'].unique().tolist()
        self.feature_names = {feature_name: 'numerical' for feature_name in temp_df.keys()[:-1]}

    def transform_car(self):
        """
        Function to transform car data set

        No major transformations are made for car

        :return self.transformed_data: DataFrame, transformed data set
        :return self.classes: int, num of classes
        """
        # We'll make a deep copy of our data set
        temp_df = pd.DataFrame.copy(self.data, deep=True)

        # Set attributes for ETL object
        self.classes = 4
        self.transformed_data = temp_df

        # Class and Feature name/type
        self.class_names = temp_df['Class'].unique().tolist()
        self.feature_names = {feature_name: 'categorical' for feature_name in temp_df.keys()[:-1]}

    def transform_segmentation(self):
        """
        Function to transform segmentation data set

        No major transformations are done for segmentation, the target class variable is ordered to the end

        :return self.transformed_data: DataFrame, transformed data set
        :return self.classes: int, num of classes
        """
        # We'll make a deep copy of our data set
        temp_df = pd.DataFrame.copy(self.data, deep=True)

        # Region pixel count is always 9 and is not useful for our algorithms
        temp_df.drop(columns='Region_Pixel_Count', inplace=True)

        # Let's reorder the class column to the back
        reordered_temp_df = temp_df.drop(columns='Class')
        reordered_temp_df['Class'] = temp_df['Class']

        # Set attributes for ETL object
        self.classes = 7
        self.transformed_data = reordered_temp_df

        # Class and Feature name/type
        self.class_names = reordered_temp_df['Class'].unique().tolist()
        self.feature_names = {feature_name: 'numerical' for feature_name in reordered_temp_df.keys()[:-1]}

    def transform_abalone(self):
        """
        Function to transform abalone data set

        No major transformations are done on this data set. The target variable is squared for percent threshold tuning

        :return self.transformed_data: DataFrame, transformed data set
        :return self.classes: int, num of classes
        """
        # We'll make a deep copy of our data set
        temp_df = pd.DataFrame.copy(self.data, deep=True)

        # Set attributes for ETL object
        self.transformed_data = temp_df

        # Feature name/type
        self.feature_names = {feature_name: 'numerical' for feature_name in temp_df.keys()[:-1]}
        self.feature_names.update({'Sex': 'categorical'})

        # Squared Average Target for percent_threshold
        self.squared_average_target = temp_df.iloc[:, -1].mean() ** 2

    def transform_machine(self):
        """
        Function to transform machine data set

        No major transformations are done on this data set, model_name is a unique ID and is removed. The target
            variable is squared for percent threshold tuning

        :return self.transformed_data: DataFrame, transformed data set
        :return self.classes: int, num of classes
        """
        # We'll make a deep copy of our data set
        temp_df = pd.DataFrame.copy(self.data, deep=True)

        # We'll remove unneeded variables as well as denormalize the target
        temp_df.drop(columns=['Model_Name', 'ERP'], inplace=True)

        # Set attributes for ETL object
        self.transformed_data = temp_df

        # Feature name/type
        self.feature_names = {feature_name: 'numerical' for feature_name in temp_df.keys()[:-1]}
        self.feature_names.update({'Vendor': 'categorical'})

        # Squared Average Target for percent_threshold
        self.squared_average_target = temp_df.iloc[:, -1].mean() ** 2

    def transform_forest_fires(self):
        """
        Function to transform forest-fires data set

        No major transformations are done on this data set. The target variable is squared for percent threshold tuning

        :return self.transformed_data: DataFrame, transformed data set
        :return self.classes: int, num of classes
        """
        # We'll make a deep copy of our data set
        temp_df = pd.DataFrame.copy(self.data, deep=True)

        # Set attributes for ETL object
        self.transformed_data = temp_df

        # Feature name/type
        self.feature_names = {feature_name: 'numerical' for feature_name in temp_df.keys()[:-1]}
        self.feature_names.update({'month': 'categorical', 'day': 'categorical'})

        # Squared Average Target for percent_threshold
        self.squared_average_target = temp_df.iloc[:, -1].mean() ** 2

    def cv_split_classification(self):
        """
        Function to split our transformed data into 10% validation and 5 cross validation splits for classification

        First this function randomizes a number between one and 10 to split out a validation set. After a number is
            randomized and the data is sorted over the class and random number. The index of the data is then mod by 5
            and each remainder represents a set for cv splitting.

        :return self.test_split: dict (of DataFrames), dictionary with keys (validation, 0, 1, 2, 3, 4) referring to the
            split transformed data
        """
        # Define base data size and size of validation
        data_size = len(self.transformed_data)
        validation_size = int(data_size / 10)

        # Check and set the random seed
        if self.random_state:
            np.random.seed(self.random_state)

        # Sample for validation
        validation_splitter = []

        # Randomize a number between 0 and 10 and multiply by the index to randomly pick observations over data set
        for index in range(validation_size):
            validation_splitter.append(np.random.choice(a=10) + (10 * index))
        self.validation_data = self.transformed_data.iloc[validation_splitter]

        # Determine the remaining index that weren't picked for validation
        remainder = list(set(self.transformed_data.index) - set(validation_splitter))
        remainder_df = pd.DataFrame(self.transformed_data.iloc[remainder]['Class'])

        # Assign a random number
        remainder_df['Random_Number'] = np.random.randint(0, len(remainder), remainder_df.shape[0])

        # Sort over class and the random number
        remainder_df.sort_values(by=['Class', 'Random_Number'], inplace=True)
        remainder_df.reset_index(inplace=True)

        # Sample for CV
        for index in range(5):
            # Mod the index by 5 and there will be 5 remainder groups for the CV split
            splitter = remainder_df.loc[remainder_df.index % 5 == index]['index']

            # Update our attribute with the dictionary for this index
            self.test_split.update({
                index: self.transformed_data.iloc[splitter]
            })

    def cv_split_regression(self):
        """
        Function to split our transformed data into 10% validation and 5 cross validation splits for regression

        First this function splits out a validation set. The remainder is sampled 5 times to produce 5 cv splits.

        :return self.test_split: dict (of DataFrames), dictionary with keys (validation, 0, 1, 2, 3, 4) referring to the
            split transformed data
        """
        # Define base data size and size of validation and splits
        data_size = len(self.transformed_data)
        validation_size = int(data_size / 10)
        cv_size = int((data_size - validation_size) / 5)

        # The extra data will go to the first splits. The remainder of the length divided by 5 defines how many extra
        extra_data = int((data_size - validation_size) % cv_size)

        # Check and set the random seed
        if self.random_state:
            np.random.seed(self.random_state)

        # Sample for validation
        validation_splitter = np.random.choice(a=data_size, size=validation_size, replace=False)
        self.validation_data = self.transformed_data.iloc[validation_splitter]

        # Determine the remaining index that weren't picked for validation
        remainder = list(set(self.transformed_data.index) - set(validation_splitter))

        # CV Split
        for index in range(5):
            # For whatever number of extra data, we'll add one to each of those index
            if (index + 1) <= extra_data:
                # Sample for the CV size if extra data
                splitter = np.random.choice(a=remainder, size=(cv_size + 1), replace=False)

            else:
                # Sample for the CV size
                splitter = np.random.choice(a=remainder, size=cv_size, replace=False)

            # Define the remaining unsampled data points
            remainder = list(set(remainder) - set(splitter))

            # Update our attribute with the dictionary for this index
            self.test_split.update({
                index: self.transformed_data.iloc[splitter]
            })

    def cv_combine(self):
        """
        Function to combine the CV splits

        For each of the 5 CV splits, this function combines the other 4 splits and assigns it the same index as the
            left out split. This combined split is labeled as the train data set
        """
        # Loop through index
        for index in range(5):
            # Remove the current index from the train_index
            train_index = [train_index for train_index in [0, 1, 2, 3, 4] if train_index != index]
            train_data = pd.DataFrame()

            # For index in our train_index, append to the Data Frame
            for data_split_index in train_index:
                train_data = train_data.append(self.test_split[data_split_index])

            # Update train data with the combined CV
            self.train_split.update({index: train_data})
