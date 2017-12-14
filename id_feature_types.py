import pandas as pd


def id_feature_types(features):

    # print('inside id_feature_types()')

    column_labels = features.columns

    feature_type_list = list()
    for column_label in column_labels:
        if features[column_label].dtype in [pd.np.dtype('float64'), pd.np.dtype('float32'), pd.np.dtype('int32'),
                                            pd.np.dtype('int32')]:
            feature_type_list.append('numeric')
        elif features[column_label].dtype == 'object':
            # print(column_label)
            features[column_label] = features[column_label].astype('category')
            feature_type_list.append('category')
        else:
            raise ValueError('unable to identify whether the response variable is a numeric or a categorical')
        # print(feature_type_list)

    list_unique_features = list(set(feature_type_list))

    length_list_unique_features = len(list_unique_features)
    if length_list_unique_features > 1:
        raise NotImplementedError('mixed numeric and categorical features - algorithm not implemented')
    elif length_list_unique_features == 1:
        feature_types = list_unique_features[0]
    else:
        raise ValueError('unable to identify type of features')

    # print('leaving id_feature_types()')

    return feature_types, features
