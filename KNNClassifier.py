class KNNClassifier(object):

    def __init__(self, minkowski_parameter, number_nearest_neighbors, response_type, feature_types,
                 response_column_label):
        import pandas as pd
        import numpy as np
        # print('instantiate knn classifier object')
        self.id = id
        self._df_train = pd.DataFrame()
        self._df_test = pd.DataFrame()
        self._response_type = response_type
        self._feature_types = feature_types
        self._response_column_label = response_column_label
        self._minkowski_parameter = minkowski_parameter
        self._number_nearest_neighbors = number_nearest_neighbors
        self._X_train = np.array([])
        self._y_train = np.array([])
        self._X_test = np.array([])
        self._y_test = np.array([])
        self._distance_table = np.array([])
        self._distance_table_test = np.array([])
        self._prediction = np.array([])
        self.accuracy_score = np.nan
        self._X_train_encode = np.array([])

    def fit(self, df):
        # print('inside .fit()')
        self._df_train = df
        if self._response_type == 'category' and self._feature_types == 'category':
            self._fit_cat_resp_cat_feat()
        elif self._response_type == 'category' and self._feature_types == 'numeric':
            self._fit_cat_resp_num_feat()
        elif self._response_type == 'numeric' and self._feature_types == 'numeric':
            raise NotImplementedError('regression for numeric response not implemented')
        elif self._response_type == 'numeric' and self._feature_types == 'category':
            raise NotImplementedError('regression for numeric response not implemented')
        else:
            raise ValueError('unrecognized combination of response and feature type')

    def _encode_cat_feat(self, X_train):
        import numpy as np
        # print('inside _encode_cat_feat()')
        from sklearn.preprocessing import LabelEncoder
        lb_make = LabelEncoder()
        X_train_encode = np.empty((X_train.shape[0], X_train.shape[1]), dtype=float)
        X_train_encode[:] = np.nan
        column_index = -1
        for column in X_train.T:
            column_index += 1
            X_train_encode[:, column_index] = lb_make.fit_transform(column).T
        return X_train, X_train_encode

    def _fit_cat_resp_cat_feat(self):
        # print('inside _fit_cat_resp_cat_feat()')
        self._set_X_and_y_numpy_arrays('train')
        self._X_train, self._X_train_encode = self._encode_cat_feat(self._X_train)

    def _scale_numerical_features(self, X):
        from sklearn.preprocessing import RobustScaler
        rBSc = RobustScaler()
        X = rBSc.fit_transform(X)
        return X

    def _construct_numerical_features_distance_table(self):
        import numpy as np
        # print('inside _construct_numerical_features_distance_table()')
        n_obs = self._X_train.shape[0]
        self._distance_table = np.empty((n_obs, n_obs), dtype=float)
        self._distance_table[:] = np.nan
        for (row, col), value in np.ndenumerate(self._distance_table):
            if col >= row:
                if col == row:
                    self._distance_table[row, col] = 0
                else:
                    row_obs = self._X_train[row, :].reshape(1, self._X_train.shape[1])
                    col_obs = self._X_train[col, :].reshape(1, self._X_train.shape[1])
                    self._distance_table[row, col] = self.distance_metric(row_obs, col_obs)
            # print(row, col, self._distance_table[row, col])

    def distance_metric(self, obs_num1, obs_num2):
        import numpy as np
        el_wise_diff = np.subtract(obs_num1, obs_num2)
        el_wise_power = np.power(el_wise_diff, self._minkowski_parameter)
        sum_el = np.sum(el_wise_power, keepdims=True)
        distance = np.power(sum_el, 1/self._minkowski_parameter)
        return distance

    def _fit_cat_resp_num_feat(self):
        # print('inside _fit_cat_resp_num_feat()')
        self._set_X_and_y_numpy_arrays('train')
        self._X_train = self._scale_numerical_features(self._X_train)
        # self._construct_numerical_features_distance_table()

    def _set_X_and_y_numpy_arrays(self, test_or_train):
        import numpy as np
        if test_or_train == 'train':
            df = self._df_train
        else:
            df = self._df_test
        n_obs = df.shape[0]
        n_feat = df.shape[1]
        X = np.array(df.iloc[0:n_obs, 1:n_feat])
        y = np.array(df.iloc[0:n_obs, 0])
        if test_or_train == 'train':
            self._X_train = X
            self._y_train = y
        else:
            self._X_test = X
            self._y_test = y

    def predict(self, df):
        # print('inside predict()')
        self._df_test = df
        if self._response_type == 'category' and self._feature_types == 'category':
            self._predict_cat_resp_cat_feat()
        elif self._response_type == 'category' and self._feature_types == 'numeric':
            self._predict_cat_resp_num_feat()
        elif self._response_type == 'numeric' and self._feature_types == 'numeric':
            raise NotImplementedError('regression for numeric response not implemented')
        elif self._response_type == 'numeric' and self._feature_types == 'category':
            raise NotImplementedError('regression for numeric response not implemented')
        else:
            raise ValueError('unrecognized combination of response and feature type')
        return self.accuracy_score, self._prediction

    def _predict_cat_resp_cat_feat(self):
        # print('inside _predict_cat_resp_cat_feat()')
        # identify nearest neighbors
        # get classification
        self._set_X_and_y_numpy_arrays('test')
        self._X_test, self._X_test_encode = self._encode_cat_feat(self._X_test)
        self._contruct_distance_table_test_cat_feat()
        self._make_prediction()

    def _contruct_distance_table_test_cat_feat(self):
        import numpy as np
        from sklearn.metrics import jaccard_similarity_score
        # print('inside _contruct_distance_table_test_cat_feat()')
        n_obs_train = self._X_train.shape[0]
        n_obs_test = self._X_test.shape[0]
        self._distance_table_test = np.empty((n_obs_test, n_obs_train), dtype=float)
        self._distance_table_test[:] = np.nan
        count = 0
        for (test_row, train_col), value in np.ndenumerate(self._distance_table_test):
            count += 1
            # print(test_row, train_col)
            if train_col < 1:
                test_row_obs = self._X_test_encode[test_row, :].reshape(1, self._X_test.shape[1])
            train_col_obs = self._X_train_encode[train_col, :].reshape(1, self._X_train.shape[1])
            jacc_sim_score = self._get_jaccard_index(test_row_obs.tolist(), train_col_obs.tolist(),
                                                     jaccard_similarity_score)
            self._distance_table_test[test_row, train_col] = 1 - jacc_sim_score

    def _get_jaccard_index(self, test_row_obs_list, train_col_obs_list, jaccard_similarity_score):
        test_row_obs_list = [item for sublist in test_row_obs_list for item in sublist]
        train_col_obs_list = [item for sublist in train_col_obs_list for item in sublist]
        jacc_sim_score = jaccard_similarity_score(test_row_obs_list, train_col_obs_list)
        return jacc_sim_score

    def _predict_cat_resp_num_feat(self):
        # print('inside _predict_cat_resp_num_feat()')
        self._set_X_and_y_numpy_arrays('test')
        self._X_test = self._scale_numerical_features(self._X_test)
        self._contruct_distance_table_test()
        self._make_prediction()

    def _make_prediction(self):
        import numpy as np
        # print('inside _make_prediction()')
        self._prediction = np.empty((self._y_test.shape[0], 1), dtype=object)
        self._prediction[:] = 'blank'
        y_train = self._y_train.reshape(1, self._y_train.shape[0])
        test_index = -1
        num_correct = 0
        num_predicted = 0
        for test_obs in self._distance_table_test:
            num_predicted += 1
            test_index += 1
            truth = self._y_test[test_index]
            nn_train_index_list = test_obs.argsort()[:self._number_nearest_neighbors]
            y_train_list = y_train[0, nn_train_index_list].tolist()
            mode = max(set(y_train_list), key=y_train_list.count)
            self._prediction[test_index, 0] = mode
            if mode == truth:
                num_correct += 1
        self.accuracy_score = num_correct/num_predicted

    def _contruct_distance_table_test(self):
        import numpy as np
        # print('inside _contruct_distance_table_test()')
        n_obs_train = self._X_train.shape[0]
        n_obs_test = self._X_test.shape[0]
        self._distance_table_test = np.empty((n_obs_test, n_obs_train), dtype=float)
        self._distance_table_test[:] = np.nan
        for (test_row, train_col), value in np.ndenumerate(self._distance_table_test):
            if train_col < 1:
                test_row_obs = self._X_test[test_row, :].reshape(1, self._X_test.shape[1])
            train_col_obs = self._X_train[train_col, :].reshape(1, self._X_train.shape[1])
            self._distance_table_test[test_row, train_col] = self.distance_metric(test_row_obs, train_col_obs)
