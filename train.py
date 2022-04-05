# %% 
# setup 

from argparse import ArgumentParser
import common

parser = ArgumentParser()
parser.add_argument('-dp', '--dataset_path', help='path to dataset')
parser.add_argument('-mp', '--model_path', help='output model filename path (defaults to model1.bin)', default='model1.bin')
parser.add_argument("--report", help="include a detailed classification report")
parser.add_argument("--confusion_matrix", help="include the confusion matrix")

# work-around for Jupyter notebook and IPython console
args = parser.parse_args()
if not args.dataset_path: 
    parser.error('dataset path not provided')
    exit(1)

parser.print_help()
print()


# %%
# dataset loading

from sklearn.model_selection import train_test_split
import numpy as np

print('loading dataset...')
x, y, entry_count = common.parse_dataset(args.dataset_path)
category_count = np.amax(y)
categories = np.arange(0, category_count)
print(f'dataset total entry count: {entry_count}')
print(f'found {category_count} categories')
print()

print('preparing dataset...')
x_train, x_test, y_train, y_test = train_test_split(x, y)
print(f'{len(x_train)} entries in train, {len(x_test)} - in test')
print()

# %% 
# feature extraction

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from time import time

def train_model(x_train, y_train, x_test, y_test):
    print("extracting features...")
    vect = TfidfVectorizer(stop_words='english', sublinear_tf=True, max_df=0.5)
    x_train = vect.fit_transform(x_train)
    print(f'train samples: {x_train.shape[0]}, features: {x_train.shape[1]}')

    x_test = vect.transform(x_test)
    print(f'test samples: {x_test.shape[0]}, features: {x_test.shape[1]}')
    print()

    feature_names = vect.get_feature_names_out()
    if feature_names.any():
        feature_names = np.asarray(feature_names)

    print('training classifier...')

    cls = RidgeClassifier()

    t0 = time() 
    cls.fit(x_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    print('testing classifier...')

    pred_result = cls.predict(x_test)

    common.calculate_metrics(y_test, pred_result, args.report, categories, args.confusion_matrix)

    return vect, cls

# %%
# save classifier and vectorizer

import pickle

def save_model(vect, cls, filepath):
    print(f'saving vectorizer and classifier to {filepath}...')

    with open(filepath, 'wb') as f:
        pickle.dump((vect, cls), f)

    print('model saved')


# %%
# putting it together

vect, cls = train_model(x_train, y_train, x_test, y_test)
print(f'vectorizer: {vect}')
print(f'classifier: {cls}')
print()

save_model(vect, cls, args.model_path)
print('all done!')
