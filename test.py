# %% 
# setup 

from argparse import ArgumentParser
import common

parser = ArgumentParser()
parser.add_argument('-dp', '--dataset_path', help='path to dataset')
parser.add_argument('-mp', '--model_path', help='input model filename path')
parser.add_argument("--report", help="include a detailed classification report")
parser.add_argument("--confusion_matrix", help="include the confusion matrix")

# work-around for Jupyter notebook and IPython console
args = parser.parse_args()
if not args.dataset_path:
    parser.error('dataset path not provided')
    exit(1)
if not args.model_path:
    parser.error('model path not provided')
    exit(1)

parser.print_help()
print()


# %%
# restore classifier and vectorizer

import pickle

def load_model(filepath):
    print(f'loading vectorizer and classifier from {filepath}...')

    with open(filepath, 'rb') as f:
        vect, cls = pickle.load(f)

    print('model loaded')

    return vect, cls


# %%
# perform test

import numpy as np
from time import time

print('loading dataset...')
x, y_test, entry_count = common.parse_dataset(args.dataset_path)
category_count = np.amax(y_test)
categories = np.arange(0, category_count)
print(f'dataset total entry count: {entry_count}')
print(f'found {category_count} categories')
print()

print('loading model...')
vect, cls = load_model(args.model_path)
print(f'vectorizer: {vect}')
print(f'classifier: {cls}')
print()

print('testing model...')
x_test = vect.transform(x)
t0 = time()
pred_result = cls.predict(x_test)
t1 = time() - t0
print("test time:  %0.3fs" % t1)
print()

common.calculate_metrics(y_test, pred_result, args.report, categories, args.confusion_matrix)

print('all done!')
print()