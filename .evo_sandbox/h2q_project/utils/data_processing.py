def clean_data(data):
    # Placeholder for data cleaning logic.
    # Currently, it only removes empty strings from list.
    if isinstance(data, list):
        return [item for item in data if item != ""]
    return data


def transform_data(data, transformation_type):
    # Placeholder for data transformation logic.
    if transformation_type == "uppercase":
        if isinstance(data, str):
            return data.upper()
        elif isinstance(data, list):
            return [item.upper() if isinstance(item, str) else item for item in data]
        else:
            return data # or raise an exception if incompatible
    return data


def validate_data(data, schema):
    # Placeholder for data validation using a schema.
    # Currently, it just checks if data is not None.
    return data is not None


class DataProcessor:
    def __init__(self, cleaning_function=clean_data, transformation_function=transform_data, validation_function=validate_data):
        self.cleaning_function = cleaning_function
        self.transformation_function = transformation_function
        self.validation_function = validation_function

    def process(self, data, transformation_type=None, schema=None):
        cleaned_data = self.cleaning_function(data)
        if transformation_type:
            transformed_data = self.transformation_function(cleaned_data, transformation_type)
        else:
            transformed_data = cleaned_data

        if schema:
            is_valid = self.validation_function(transformed_data, schema)
            if not is_valid:
                raise ValueError("Data validation failed.")
        return transformed_data