import unittest
import pandas as pd
from h2q_project.data_loader import DataLoader
import os

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # Create a dummy CSV file for testing
        self.test_file = 'test_data.csv'
        data = {'col1': [1, 2, 3, 4, 5], 'col2': [6, 7, 8, 9, 10]}
        self.df = pd.DataFrame(data)
        self.df.to_csv(self.test_file, index=False)

    def tearDown(self):
        # Remove the dummy CSV file after testing
        os.remove(self.test_file)

    def test_load_data(self):
        loader = DataLoader(self.test_file)
        loaded_df = loader.load_data(chunksize=2)
        pd.testing.assert_frame_equal(loaded_df, self.df)

    def test_load_data_empty_file(self):
        # Create an empty CSV file
        empty_file = 'empty_data.csv'
        open(empty_file, 'w').close() # Creates an empty file
        
        loader = DataLoader(empty_file)
        loaded_df = loader.load_data(chunksize=2)
        
        #Expected behaviour: should return an empty dataframe without errors
        self.assertTrue(loaded_df.empty)
        os.remove(empty_file)


if __name__ == '__main__':
    unittest.main()