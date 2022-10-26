import argparse


# argument parsing
# parser = argparse.ArgumentParser(description='Botnet Feature Selection and Models ')

def command_line_arguments():
    """

    :return: parsing strings
    """
    parser = argparse.ArgumentParser(description='models')
    # parser.add_argument('-cv', '--cv_score', type=int, metavar='',help='Cv score should be integer')
    parser.add_argument('-f', '--filename', type=str, metavar='', required=True, help='data file name')
    parser.add_argument('-d', '--foldername', type=str, metavar='', required=True, help='data folder path')
    parser.add_argument('-S', '--sample', type=int, metavar='', required=True, help='Data Sample Size')
    parser.add_argument('-fn', '--fnumber', type=int, metavar='', required=True,help='Feature numbers like N-BaIoT dataset has 115, Med-Biot has 100 features. so fn= 100 or 115')
    parser.add_argument('-c', '--class_name', type=str, metavar='', required=True,help=' class name ,class name should be "class-1, class-2, class-3"')
    parser.add_argument('-m', '--metric', type=str, metavar='',help=' accuracy, f1_macro, recall_macro, precision_macro   ')
    # # parser.add_argument( '-d','--direction',type=bool,metavar='',help='direction is forward and backward. \nfor the Forward direction it is True , Backward direction it is False ')
    # # parser.add_argument('-f','--floating',type=bool,metavar='',help='Floating is the option of sequential feature selection. \nif you want floating please keep --True or else Fasle')
    # parser.add_argument('-op', '--out', type=str, metavar='',
    #                     help='output option is the storing the file for the fitting method')
    # # parser.add_argument('-2','--binary',type=str,metavar='',help='binary class object')
    # # parser.add_argument('-3','--3_class',type=str,metavar='',help='3_class')
    # # parser.add_argument('-9','--9_class',type=str,metavar='',help='')
    parser.add_argument('-rp', '--rpath', type=str, metavar='', help='give the resultant path  saving the  model')
    # parser.add_argument('-md', '--modename', type=str, metavar='',
    #                     help='give strings  like xgb, lgb,  et, gb, rf, dt, rf')
    # parser.add_argument('-ds', '--dname', type=str, metavar='', help='data set names are nbiot and Med-biot')
    parser.add_argument('-ptype', '--paramtype', type=str, metavar='', help='Hyper Parameter Tuning type name.')
    parser.add_argument('-fname', '--fsname', type=str, metavar='Feature Selection name type are 1. fisher_score, 2. mutual_info', help='Feature selection ')
    parser.add_argument('-fcount', '--fscount', type=int, metavar='Number of Features', help='number feature are taken by feature selection method name')
    return parser
