from sklearn.preprocessing import MinMaxScaler
import sklearn.svm as svm
from sklearn.utils import shuffle


def svm_model(input_matrix, input_classes, C=16, kernel='rbf', gamma=0.25, tol=0.0000001):
	input_matrix, input_classes = shuffle(input_matrix, input_classes)
	classifier = svm.SVC(C = C, kernel=kernel, gamma=gamma, tol=tol, class_weight="balanced", verbose=False)
	scaler = MinMaxScaler((-1, 1))
	scaled_X = scaler.fit_transform(input_matrix)
	model = classifier.fit(scaled_X, input_classes)
	return (model, scaler)

def test_model(test_matrix, test_classes, model, scaler):
	scaled_X = scaler.transform(test_matrix)
	predicted_classes = model.predict(scaled_X)
	return(predicted_classes)