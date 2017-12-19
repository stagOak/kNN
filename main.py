# from naive_bayes_glue_layer import naive_bayes_glue_layer
from load_data import load_data
from format_data_frame_and_id_feature_types import format_data_frame_and_id_feature_types
from make_response_column_first_column import make_response_column_first_column
from id_response_type import id_response_type
from id_feature_types import id_feature_types
from split_train_test import split_train_test
from knn_classifier_glue_layer import knn_classifier_glue_layer
from sklearn.metrics import confusion_matrix

# data file options

file_path_and_name = 'Iris.csv'
response_column_label = 'Species'

# file_path_and_name = 'miniTennis.csv'
# response_column_label = 'Species'
#
# file_path_and_name = 'Titanic.csv'
# response_column_label = 'Survived'

# load data and determine feature type
df = load_data(file_path_and_name, print_verbose=False)
df, response_type, feature_types = format_data_frame_and_id_feature_types(df, response_column_label,
                                                                          make_response_column_first_column,
                                                                          id_response_type, id_feature_types,
                                                                          print_verbose=False)
# print('\n\n\n******')
# print('df')
# print()
# print(df.info())
# print()
# print(df.head())
# print(response_type)
# print(feature_types)
# print('******')

df_train, df_test = split_train_test(df, train_split_fraction=0.7, set_seed=True, print_verbose=False)

# print('\n\n\n******')
# print('df_train')
# print()
# print(df_train.info())
# print()
# print(df_train.head())
# print()
# print('df_test')
# print()
# print(df_test.info())
# print()
# print(df_test.head())
# print('******')

# in this assignment the kNN classifier is implemented as a class called KNNClassifier
# a kNN regressor is not implemented
# the knn_classifier_glue_layer below bridges the assignment requirements and the object oriented design

minkowski_parameter = 2
number_nearest_neighbors = 5

accuracy_score, classification = knn_classifier_glue_layer(df_train, df_test, response_type, feature_types,
                                                           response_column_label, minkowski_parameter,
                                                           number_nearest_neighbors)

print()
print('classification prediction:')
print(classification)
print()
print('truth')
print('classification truth:')
print(df_test.iloc[:, 0])
print()
print('accuracy_score = ', accuracy_score)
print()
print('confusion matrix: rows are truth, columns are prediction:')
print(confusion_matrix(df_test.iloc[:, 0], classification))
