def make_response_column_first_column(df, response_column_label, print_verbose):

    list_of_column_labels = list(df)
    ordered_list_of_column_labels = list()
    for column_label in list_of_column_labels:
        if column_label == response_column_label:
            ordered_list_of_column_labels.append(column_label)
            list_of_column_labels.remove(column_label)
            break

    ordered_list_of_column_labels.extend(list_of_column_labels)
    df_formatted = df[ordered_list_of_column_labels].copy()

    if print_verbose:
        print('\n*******')
        print('make_response_column_first_column()')
        print()
        print('unformatted data frame')
        print(df.head())
        print()
        print('formatted data frame')
        print('response_column_label = ', response_column_label, ' ; should be first column')
        print()
        print(df_formatted.head())
        print('\n*******')

    return df_formatted, 'numeric'
