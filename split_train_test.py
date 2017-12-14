import pandas as pd
import numpy as np


def split_train_test(df, train_split_fraction, set_seed, print_verbose):

    if set_seed:  # for shuffle to be reproducible
        np.random.seed(0)

    # shuffle the data frame
    df = df.sample(frac=1).reset_index(drop=True)

    if set_seed:  # for train/test split to be reproducible
        np.random.seed(0)

    msk = np.random.rand(len(df)) < train_split_fraction
    df_train = df[msk].reset_index(drop=True)
    df_test = df[~msk].reset_index(drop=True)

    if print_verbose:
        print('df.shape = ', df.shape)
        print('\n*******')
        print('df_train.shape = ', df_train.shape)
        print('\n*******')
        print('df_test.shape = ', df_test.shape)

    return df_train, df_test
