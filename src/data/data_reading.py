import pandas as pd
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


def datareading(folder_name, file_name, class_name, sample_size, x_numbers):
    """
    Data reading folder of data reading and taking the samples
    :param folder_name: data reading Folder,
    :param file_name: data file name, .csv file
    :param class_name:  give the class name
    :param sample_size: Sample of the dataset.
    :param x_numbers: if N-BaIoT data set give 115, or if Med-IoT dataset is 100.
    :return: X(pandas.core.frame.DataFrame), y(pandas.core.frame.DataFrame).
    """
    print("==" * 40)
    data = pd.read_csv(Path().joinpath(folder_name, file_name))
    class_name = data[class_name].name
    if len(data[class_name].unique()) == 2:
        print("Binary Classification.")
    else:
        print("Multi-Class Classification.")
    shape = data.shape
    print('shape:{0}'.format(str(shape)))
    print(f"class_name:{class_name}")
    print("class Labels:{0}".format(str(data[class_name].unique())))
    df = data.groupby(class_name).apply(lambda x: x.sample(n=sample_size)).reset_index(drop=True)
    # Encoding the labels.
    le = LabelEncoder()
    cols = df.columns.to_list()
    for column in cols:
        if df[column].name == class_name:
            df[column] = le.fit_transform(df[column])
    # data samples
    X = df.iloc[:, 0:x_numbers]
    y = df[class_name]
    print("==" * 40)
    return X, y





