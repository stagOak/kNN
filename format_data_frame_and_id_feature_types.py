def format_data_frame_and_id_feature_types(df, response_column_label, make_response_column_first_column,
                                           id_response_type, id_feature_types, print_verbose):

    df, response_type = make_response_column_first_column(df, response_column_label, print_verbose)

    response_type, df.iloc[:, 0] = id_response_type(df.iloc[:, 0])
    # print(response_type)
    # print(df.info())

    feature_types, df.iloc[:, 1:] = id_feature_types(df.iloc[:, 1:])
    # print(feature_types)
    # print(df.info())

    return df, response_type, feature_types
