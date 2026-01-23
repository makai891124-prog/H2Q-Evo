import pytest
from h2q_project.validation import validate_data


def test_validate_data_valid():
    data = {"name": "John Doe", "age": 30}
    assert validate_data(data) == True


def test_validate_data_invalid_type():
    data = "invalid"
    assert validate_data(data) == False


def test_validate_data_missing_name():
    data = {"age": 30}
    assert validate_data(data) == False


def test_validate_data_missing_age():
    data = {"name": "John Doe"}
    assert validate_data(data) == False


def test_validate_data_invalid_age():
    data = {"name": "John Doe", "age": -1}
    assert validate_data(data) == False


def test_validate_data_invalid_name_type():
    data = {"name": 123, "age": 30}
    assert validate_data(data) == False


def test_validate_data_invalid_age_type():
    data = {"name": "John Doe", "age": "30"}
    assert validate_data(data) == False