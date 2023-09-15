import random
import numpy as np
import torch

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything()

import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from deepctr_torch.inputs import SparseFeat, get_feature_names
from deepctr_torch.models import DeepFM as DeepFM

fname = 'pred_ans_DeepFM.npy'

if __name__ == "__main__":

    train_data = pd.read_csv("./train_set.csv", sep=',')
    test_data = pd.read_csv("./test_set.csv", sep=',')
    valid_data = pd.read_csv("./valid_set.csv", sep=',')

    data = pd.concat([train_data, test_data, valid_data], axis=0)

    sparse_features = ["movie_id", "user_id",
                       "gender", "age", "occupation", "zipcode",
                       'Name', 'Year', 'genre1',  'genre2', 'genre3', 'genre4', 'genre5', 'genre6', 'genre7', 'genre8', 'genre9', 'genre10', 'genre11', 'genre12', 'genre13', 'genre14', 'genre15', 'genre16', 'genre17', 'genre18','genre19',
                       'director','writers','stars'
                       ]
    target = ['rating']

    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])

    train_data = pd.DataFrame(data[:len(train_data)])
    test_data = pd.DataFrame(data[len(train_data):len(train_data)+len(test_data)])
    valid_data = pd.DataFrame(data[len(train_data)+len(test_data):])

    fixlen_feature_columns = [SparseFeat(feat, data[feat].nunique())
                            for feat in sparse_features]
    linear_feature_columns = fixlen_feature_columns
    dnn_feature_columns = fixlen_feature_columns
    feature_names = get_feature_names(linear_feature_columns + dnn_feature_columns)

    train_model_input = {name: train_data[name] for name in feature_names}
    test_model_input = {name: test_data[name] for name in feature_names}
    valid_model_input = {name: valid_data[name] for name in feature_names}

    device = 'cpu'
    use_cuda = True
    if use_cuda and torch.cuda.is_available():
        print('cuda ready...')
        device = 'cuda:0'

    model = DeepFM(linear_feature_columns, dnn_feature_columns, task='regression', device=device)
    model.compile("adam", "mse", metrics=['mse'], )

    history = model.fit(train_model_input,train_data[target].values,batch_size=256,epochs=50,verbose=2,validation_data=(valid_model_input,valid_data[target].values))
    pred_ans = model.predict(test_model_input, batch_size=256)

    np.save(fname, pred_ans)
    np.save('true_ans.npy', test_data[target].values)

    print("\nRMSE:", round(np.sqrt(mean_squared_error(test_data[target].values, pred_ans)),4), '\nMAE:',round(mean_absolute_error(test_data[target].values, pred_ans),4))

