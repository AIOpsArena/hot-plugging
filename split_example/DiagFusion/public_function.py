import os
import pickle
import argparse
import yaml
import pathlib
DF_path = pathlib.Path(__file__).parent

def load(file):
    with open(file, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    return data


def save(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)


def min_max_normalized(feature):
    feature_copy = feature.copy().astype(float)
    for i in range(len(feature_copy)):
        min_f, max_f = min(feature_copy[i]), max(feature_copy[i])
        if min_f == max_f:
            feature_copy[i] = [0]*len(feature_copy[i])
        else:
            feature_copy[i] = (feature_copy[i] - min_f) / (max_f - min_f)
    return feature_copy

