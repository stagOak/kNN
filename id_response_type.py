import pandas as pd


def id_response_type(response):

    # print('inside id_response_type()')

    if response.dtype in [pd.np.dtype('float64'), pd.np.dtype('float32'), pd.np.dtype('int32'), pd.np.dtype('int32')]:
        raise NotImplementedError('Algorithm has not been implemented for numeric response.')
        # response_type = 'numeric'
    elif response.dtype == 'object':
        response = response.astype('category')
        response_type = 'category'
    else:
        raise ValueError('unable to identify whether the response variable is a numeric or a categorical')

    # print('leaving id_response_type()')

    return response_type, response
