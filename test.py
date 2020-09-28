import argparse
import os
import h5py
import numpy as np
from keras.models import load_model
import pandas as pd
from keras.models import model_from_json
import json

def loaddata(data_dir, filename):

    with h5py.File(data_dir+filename, 'r') as f:
        # List all groups
        a_group_key = list(f.keys())[0]
        # Get the data
        data = list(f[a_group_key])
    X= np.asarray(data, 'float32')

    return X

def main():
    parser = argparse.ArgumentParser(
        description='test AD recognition')
    parser.add_argument('--input', type=str, required=True, help="path to test data")
    parser.add_argument('--model', type=str, required=True, help="path to pre-trained model")
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()

    model_dir = os.path.join(os.path.dirname(os.getcwd()), args.model)
    A = loaddata(args.input, 'testa.h5')
    B = loaddata(args.input, 'testb.h5')
    print('A_shape:{}'.format(A.shape))
    print('B_shape:{}'.format(B.shape))
    print("[INFO] loading pre-trained network...")
    json_file = open(model_dir+'fold3.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    # load weights into new model
    model.load_weights(model_dir+"fold3.hd5")

    pred_la= []
    testa_id= []
    pred_lb = []
    testb_id = []
    for i in range(A.shape[0]):
        pred = model.predict(np.expand_dims(A[i], axis=0))
        pred = pred.argmax(axis=1)[0]
        pred_la.append(pred)
        testa_id.append('testa_{}'.format(i))

    for i in range(B.shape[0]):
        pred = model.predict(np.expand_dims(B[i], axis=0))
        pred = pred.argmax(axis=1)[0]
        pred_lb.append(pred)
        testb_id.append('testb_{}'.format(i))

    df = pd.DataFrame({'testa_id':testa_id+testb_id, 'label':pred_la+pred_lb})
    df.to_csv(os.path.join(args.output, 'test_result.csv'), index=False)


if __name__ == '__main__':
    main()
