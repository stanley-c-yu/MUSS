from sklearn.preprocessing import MinMaxScaler
import sklearn.svm as svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, f1_score
import numpy as np


def svm_model(input_matrix,
              input_classes,
              C=16,
              kernel='rbf',
              # 5 better than 0.25, especially between ambulation and lying, we should use a larger gamma for higher decaying impact of neighbor samples.
              gamma=0.25,
              tol=0.0001):
    input_matrix, input_classes = shuffle(input_matrix, input_classes)
    classifier = svm.SVC(
        C=C,
        kernel=kernel,
        gamma=gamma,
        tol=tol,
        probability=False,
        class_weight='balanced')
    # verbose=False)
    # classifier = RandomForestClassifier(
    #     n_estimators=100, max_depth=None, min_samples_split=2, random_state=0, n_jobs=-1)
    scaler = MinMaxScaler((-1, 1))
    scaled_X = scaler.fit_transform(input_matrix)
    model = classifier.fit(scaled_X, input_classes)
    train_accuracy = model.score(scaled_X, input_classes)
    return (model, scaler, train_accuracy)


def test_model(test_matrix, test_classes, model, scaler):
    scaled_X = scaler.transform(test_matrix)
    predicted_classes = model.predict(scaled_X)
    return (predicted_classes)


def loso_validation(input_matrix, input_classes, groups, **model_kwargs):
    loso = LeaveOneGroupOut()
    output_predictions = np.copy(input_classes)
    for train_indices, test_indices in loso.split(
            input_matrix, input_classes, groups=groups):
        train_set = input_matrix[train_indices, :]
        train_labels = input_classes[train_indices]
        test_set = input_matrix[test_indices, :]
        test_labels = input_classes[test_indices]
        model, scaler, _ = svm_model(train_set, train_labels, **model_kwargs)
        predicted_labels = test_model(test_set, test_labels, model, scaler)
        output_predictions[test_indices] = predicted_labels
    metric = f1_score(input_classes, output_predictions, average='macro')
    return output_predictions, metric
