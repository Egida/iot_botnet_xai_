import argparse
import sys
from src.models.model_fitting import ModelFittingPipeLine
from src.data.data_reading import datareading
from src.command_line_parsing import command_line_arguments
import joblib
from sys import path

REQUIRED_PYTHON = "python3"


def main():
    """
    python version checking.
    """
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


parser = command_line_arguments()
args = parser.parse_args()
folder_name = args.foldername
print("folder names".format(folder_name))

if __name__ == '__main__':
    main()

    directory_path = '/gpfs/mariana/home/rkalak/xai_evaluation/project/Iot_Botnet_XAI'

    sys.path.insert(0, directory_path)

    X, y = datareading(args.foldername, args.filename, args.class_name, sample_size=args.sample,x_numbers=args.fnumber)
    #
    # print(X.shape, y.shape)
    # X, y, metric_type,
    # feature_selection_method_name = 'fisher_score',
    # number_of_features = 5,
    # search_type = 'grid_search',
    # file_location = ""

    # result_path = args.rpath.strip()
    # file_path = args.filename
    # print("File path: {0}/{1}".format(result_path, file_path))

    # print("resultant  path:{0}".format(result_path))
    pipe_line = ModelFittingPipeLine(X, y,args.metric,args.fsname,args.fscount, args.paramtype,args.rpath)

    res = pipe_line.fitting_models()



    # joblib.dump(res,args.path/'model_res.pkl')
