from h2q_project.validation import validate_data

def process_data(data):
    if validate_data(data):
        print("Data is valid")
    else:
        print("Data is invalid")


if __name__ == '__main__':
    data1 = {"name": "Alice", "age": 25}
    data2 = {"name": "Bob", "age": -5}
    process_data(data1)
    process_data(data2)