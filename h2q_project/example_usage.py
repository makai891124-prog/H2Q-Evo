from h2q_project.data_loader import DataLoader

# Example usage of the DataLoader

def main():
    file_path = 'h2q_project/data.jsonl' # Assuming data is in JSON Lines format
    data_loader = DataLoader(file_path)

    print("Streaming data loading:")
    for item in data_loader.stream_load_data():
        if item:
            print(f"Processing item: {item}")
        else:
            print("No more data or an error occurred.")
            break

    print("\nLoading all data at once (for comparison or small datasets):")
    all_data = data_loader.load_all_data()
    if all_data:
        print(f"All data loaded: {all_data}")
    else:
        print("Failed to load all data.")

if __name__ == "__main__":
    main()