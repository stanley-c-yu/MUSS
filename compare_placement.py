import pandas as pd
import os
from helper.svm_model import svm_model, test_model
from sklearn.metrics import classification_report, f1_score
import seaborn as sns
import matplotlib.pyplot as plt


def compare_placement(joined_set, target, train_placement, test_placement):
    train_set = joined_set.loc[joined_set['GROUP2'] == train_placement,:]
    test_set = joined_set.loc[joined_set['GROUP2'] == test_placement,:]
    
    cols = [0, 1,2] + list(range(5,21)) + [24]
    train_data = train_set[train_set.columns[cols]].set_index(['START_TIME', 'STOP_TIME', 'GROUP0'])
    test_data = test_set[test_set.columns[cols]].set_index(['START_TIME', 'STOP_TIME', 'GROUP0'])

    model, scaler = svm_model(train_data.values[:,:-1], train_data.values[:,-1])
    predictions = test_model(test_data.values[:,:-1], test_data.values[:,-1], model, scaler)
    
    test_set.loc[:,'PREDICTION'] = predictions
    return test_set
    

if __name__ == '__main__':
    input_folder = 'D:/data/spades_lab'
    output_folder = os.path.join(
        input_folder, 'DerivedCrossParticipants', 'location_matters')
    joined_set_file = os.path.join(output_folder, 'SPADES_1_S_None_train.csv')
    joined_set = pd.read_csv(joined_set_file, parse_dates=[0,1], infer_datetime_format=True)

    placements = list(filter(lambda name: 'wear' not in name, joined_set['GROUP2'].unique()))

    f1_scores = []
    for train_placement in placements:
        for test_placement in placements:
            print('train: ' + train_placement)
            print('test: ' + test_placement)
            prediction_set = compare_placement(joined_set, target='posture', train_placement=train_placement, test_placement=test_placement)
            y_true = prediction_set['POSTURE'].values
            y_pred = prediction_set['PREDICTION'].values
            score = f1_score(y_true, y_pred, average='weighted')
            f1_scores.append(pd.DataFrame(data=[[train_placement, test_placement, score]], columns=['TRAIN', 'TEST', 'F1_SCORE'], index=[0]))
    f1_scores = pd.concat(f1_scores)
    map_scores = f1_scores.pivot(index='TRAIN', columns='TEST', values='F1_SCORE')
    map_scores.to_csv(os.path.join(output_folder, 'compare_placement_posture_single.csv'), index=True)
    p = sns.heatmap(map_scores, annot=True, vmin=0, vmax=1, cmap="YlGnBu")
    