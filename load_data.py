import pandas as pd


def load_data(file_path_and_name, print_verbose):

    df = pd.read_csv(file_path_and_name)

    if print_verbose:
        print('\n*******')
        print('inside load_data()')
        print()
        print('data frame loaded (df):')
        print(df.head())
        print('df.shape = ', df.shape)
        print('\n*******')

    return df
