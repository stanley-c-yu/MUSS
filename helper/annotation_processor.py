#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 11:03:13 2018

@author: zhangzhanming
"""

import pandas as pd
import sys
import os.path
import re
import collections
from .semantic_similarity import get_most_similar, get_most_similar_batch
from .spadeslab_class_labeler import to_activity_spadeslab


def annotation_splitter(in_annotation):
    '''
    Combine overlapped labels and split them according to time

    :param pandas.DataFrame in_annotation: the raw annotation to split
    '''

    time_list = []
    # iterate through the annotation data, put (start time/end time, label name
    # start/end, index) to a list
    for index, series in in_annotation.iterrows():
        time_list.append((pd.to_datetime(series[1]),
                          series[3], 'start', index))
        time_list.append((pd.to_datetime(series[2]),
                          series[3], 'end', index))
    time_list.sort(key=lambda tup: tup[0])  # sort the list according to time

    # iterate through the time list, detect overlap. If exist, split and concatenate them
    curr_activities = []
    splitted_time_list = []
    last_time = time_list[0][0]

    for time_record in time_list:
        curr_time = time_record[0]
        if curr_time == last_time and time_record[2] == 'start':
            curr_activities.append(time_record[1].lower().strip())
        else:
            if len(curr_activities) > 0 and last_time != curr_time:
                curr_activities.sort()
                new_label = ' '.join(curr_activities)
                splitted_time_list.append(pd.Series([last_time, last_time, curr_time, new_label],
                                                    index=['HEADER_TIME_STAMP', 'START_TIME', 'STOP_TIME', 'LABEL_NAME']))
            if time_record[2] == 'start':
                curr_activities.append(time_record[1].lower().strip())
            else:
                curr_activities.remove(time_record[1].lower().strip())

            last_time = curr_time

    # sort by start time, export to dataframe
    splitted_time_list.sort(key=lambda series: series['START_TIME'])
    splitted_annotation = pd.DataFrame(splitted_time_list)

    return splitted_annotation


class ClassLabeler:
    def __init__(self, class_label_set):
        if isinstance(class_label_set, str):
            self._class_labels = pd.read_csv(class_label_set)
        else:
            self._class_labels = class_label_set
        self._primary_class_labels = self._get_primary_class_labels()

    def _get_primary_class_labels(self):
        unique_classes = self._class_labels.nunique()
        col_name = unique_classes[unique_classes ==
                                  max(unique_classes)].index[0]
        return self._class_labels.loc[:, col_name]

    def from_annotation_labels(self, labels):
        print("Getting class labels for: " + labels)
        label_str = labels
        # matched_primary_class_label = get_most_similar(
            # label_str, self._primary_class_labels.values)
        matched_primary_class_label = to_activity_spadeslab(label_str)
        return self.from_primary_class_label(matched_primary_class_label, label_str)

    def from_primary_class_label(self, primary_class_label, label_str):
        matched_class_labels = self._class_labels.loc[primary_class_label ==
                                                      self._primary_class_labels.values, :]
        matched_class_labels.insert(0, 'ANNOTATION_LABELS', [label_str])
        return matched_class_labels

    def from_annotation_labels_list(self, labels_list):
        # matched_primary_class_label_list = get_most_similar_batch(
            # labels_list, self._primary_class_labels.values)
        matched_primary_class_label_list = [to_activity_spadeslab(labels) for labels in labels_list]
        return pd.concat([self.from_primary_class_label(matched_primary_class_label, label_str) for matched_primary_class_label, label_str in zip(matched_primary_class_label_list, labels_list)])

    @staticmethod
    def from_annotation_set(annotations, class_label_set, interval):
        durations = annotations.iloc[:, 2] - annotations.iloc[:, 1]
        valid_annotations = annotations.loc[durations > pd.Timedelta(
            interval, unit='s'), :]
        print('in total annotations: ' + str(valid_annotations.shape[0]))
        class_labeler = ClassLabeler(class_label_set=class_label_set)
        annotation_set = valid_annotations.iloc[:, 3].unique()
        class_label_map = class_labeler.from_annotation_labels_list(
            annotation_set)
        return class_label_map


if __name__ == '__main__':
    class_labeler = ClassLabeler(
        class_label_set='C:/Users/tqshe/Projects/python/location_matters/data/location_matters.csv')
    matched_label = class_labeler.from_annotation_labels(
        labels=['standing sweeping telling story'])
    print(matched_label)
