import pandas as pd

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None

    def load_data(self):
        try:
            # Use pandas for efficient CSV loading, handle potential errors gracefully
            self.data = pd.read_csv(self.file_path)
            print(f"Data loaded successfully from {self.file_path}")
        except FileNotFoundError:
            print(f"Error: File not found at {self.file_path}")
            self.data = None  # Ensure data is None in case of an error
        except Exception as e:
            print(f"An error occurred during data loading: {e}")
            self.data = None
        return self.data

    def process_data(self):
        if self.data is None:
            print("No data to process. Please load data first.")
            return None

        # Example processing: Convert column names to lowercase
        self.data.columns = [col.lower() for col in self.data.columns]

        # Example processing: Handle missing values by filling with the mean
        self.data = self.data.fillna(self.data.mean(numeric_only=True))

        print("Data processing complete.")
        return self.data

    def get_data(self):
        return self.data


if __name__ == '__main__':
    # Example usage (replace with your actual file path)
    file_path = 'h2q_project/sample_data.csv'  # Relative path within h2q_project
    data_loader = DataLoader(file_path)
    data = data_loader.load_data()

    if data is not None:
        processed_data = data_loader.process_data()
        if processed_data is not None:
            print(processed_data.head())
