#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 11:03:13 2018

@author: zhangzhanming
"""

import pandas as pd
from padar_converter.dataset import spades
import os


def get_pa_abbr_labels(dataset_folder):
    filepath = os.path.join(dataset_folder, 'MetaCrossParticipants',
                            'muss_class_labels.csv')
    label_mapping = pd.read_csv(filepath)
    labels = label_mapping['MUSS_22_ACTIVITY_ABBRS'].unique().tolist()
    labels.remove('Unknown')
    labels.remove('Transit.')
    return labels


def get_pa_labels(dataset_folder):
    filepath = os.path.join(dataset_folder, 'MetaCrossParticipants',
                            'muss_class_labels.csv')
    label_mapping = pd.read_csv(filepath)
    labels = label_mapping['MUSS_22_ACTIVITIES'].unique().tolist()
    labels.remove('Unknown')
    labels.remove('Transition')
    return labels


class ClassLabeler:
    def __init__(self, class_label_set):
        if isinstance(class_label_set, str):
            self._class_labels = pd.read_csv(class_label_set)
        else:
            self._class_labels = class_label_set
        self._primary_class_labels = self._get_primary_class_labels()

    def _get_primary_class_labels(self):
        unique_classes = self._class_labels.nunique()
        col_name = unique_classes[unique_classes == max(
            unique_classes)].index[0]
        return self._class_labels.loc[:, col_name]

    def from_annotation_labels(self, labels):
        print("Getting class labels for: " + labels)
        label_str = labels
        matched_primary_class_label = spades.to_inlab_activity_labels(
            label_str)
        return self.from_primary_class_label(matched_primary_class_label,
                                             label_str)

    def from_primary_class_label(self, primary_class_label, label_str):
        matched_class_labels = self._class_labels.loc[
            primary_class_label == self._primary_class_labels.values, :]
        matched_class_labels.insert(0, 'ANNOTATION_LABELS', [label_str])
        return matched_class_labels

    def from_annotation_labels_list(self, labels_list):
        matched_primary_class_label_list = [
            spades.to_inlab_activity_labels(labels) for labels in labels_list
        ]
        result = pd.concat([
            self.from_primary_class_label(
                matched_primary_class_label,
                label_str.lower().replace('wear on', '').replace('wearon',
                                                                 '').strip())
            for matched_primary_class_label, label_str in zip(
                matched_primary_class_label_list, labels_list)
        ])
        result = result.drop_duplicates()
        return result

    @staticmethod
    def from_annotation_set(annotations, class_label_set, interval):
        durations = annotations.iloc[:, 2] - annotations.iloc[:, 1]
        valid_annotations = annotations.loc[durations > pd.
                                            Timedelta(interval, unit='s'), :]
        valid_annotations = valid_annotations.dropna()
        print('in total annotations: ' + str(valid_annotations.shape[0]))
        class_labeler = ClassLabeler(class_label_set=class_label_set)
        annotation_set = valid_annotations.iloc[:, 3].unique()
        class_label_map = class_labeler.from_annotation_labels_list(
            annotation_set)
        return class_label_map
