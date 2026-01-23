import unittest
from h2q_project.core.data_processing import clean_data, load_data
import pandas as pd

class TestDataProcessing(unittest.TestCase):

    def test_load_data(self):
        # Create a dummy csv file for testing
        test_csv_content = "col1,col2\n1,2\n3,4"
        with open("test_data.csv", "w") as f:
            f.write(test_csv_content)

        # Load the data using the load_data function
        df = load_data("test_data.csv")

        # Assert that the loaded data is a pandas DataFrame
        self.assertIsInstance(df, pd.DataFrame)

        # Assert that the DataFrame has the correct shape
        self.assertEqual(df.shape, (2, 2))

        # Assert that the DataFrame contains the correct data
        self.assertEqual(df['col1'].tolist(), [1, 3])
        self.assertEqual(df['col2'].tolist(), [2, 4])

    def test_clean_data(self):
        # Create a sample DataFrame with missing and duplicate values
        data = {'col1': [1, 2, None, 4, 2], 'col2': [5, 6, 7, 8, 6]}
        df = pd.DataFrame(data)

        # Clean the data using the clean_data function
        cleaned_df = clean_data(df)

        # Assert that missing values have been dropped
        self.assertFalse(cleaned_df.isnull().values.any())

        # Assert that duplicate values have been dropped
        self.assertEqual(len(cleaned_df), 3)

if __name__ == '__main__':
    unittest.main()