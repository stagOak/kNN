from KNNClassifier import KNNClassifier


def knn_classifier_glue_layer(df_train, df_test, response_type, feature_types, response_column_label,
                              minkowski_parameter, number_nearest_neighbors):

    # print('inside knn_classifier_glue_layer()')

    # instantiate knn classifier object, fit and predict
    knn = KNNClassifier(minkowski_parameter, number_nearest_neighbors, response_type, feature_types,
                        response_column_label)
    knn.fit(df_train)
    accuracy_score, prediction = knn.predict(df_test)

    return accuracy_score, prediction
