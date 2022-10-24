import sys
from src.models.model_fitting import ModelFittingPipeLine
from src.data.data_reading import datareading

REQUIRED_PYTHON = "python3"


def main():
    system_major = sys.version_info.major
    if REQUIRED_PYTHON == "python":
        required_major = 2
    elif REQUIRED_PYTHON == "python3":
        required_major = 3
    else:
        raise ValueError("Unrecognized python interpreter: {}".format(
            REQUIRED_PYTHON))

    if system_major != required_major:
        raise TypeError(
            "This project requires Python {}. Found: Python {}".format(
                required_major, sys.version))
    else:
        print(">>> Development environment passes all tests!")



if __name__ == '__main__':
    main()
    folder_name = 'data/processed'
    X, y = datareading(folder_name=folder_name, file_name='data_file.csv', class_name='class-1', sample_size=50,
                       x_numbers=115)

    