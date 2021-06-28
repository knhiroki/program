import numpy as np
import pandas as pd
import os
import glob
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def z_score(x, axis=None):
    x_mean = x.mean(axis=axis, keepdims=True)
    x_std = x.std(axis=axis, keepdims=True)
    x_new = (x - x_mean) / x_std

    return x_new


def min_max(x, axis=None):
    x_min = x.min(axis=axis, keepdims=True)
    x_max = x.max(axis=axis, keepdims=True)
    result = (x - x_min) / (x_max - x_min)
    return result


class Identification:

    def __init__(self, start_frequency, end_frequency, target):

        self.start_frequency = start_frequency
        self.end_frequency = end_frequency
        self.target = target
        self.train_f = np.empty(0)
        self.train_label = np.empty(0)
        self.train_name = np.empty(0)
        self.test_f = np.empty(0)
        self.test_label = np.empty(0)
        self.test_name = np.empty(0)

    def labeling(self):

        np.set_printoptions(threshold=np.inf)

        frequency_range = str(self.start_frequency) + '<=' + 'frequency' + '<=' + str(self.end_frequency)

        for i, name in enumerate(self.target, 0):
            path = os.path.join('C:\\Users\\shira\\data_folder\\training_data', name)
            path = os.path.join(path, 'Trans_*.csv')
            train_path = sorted(glob.glob(path))
            for j in range(0, len(train_path)):
                t_f = pd.read_csv(train_path[j], header=None, names=['frequency', 'transmittance'])
                t_name = os.path.basename(train_path[j])
                self.train_name = np.append(self.train_name, t_name)
                t_f = t_f.query(frequency_range)
                t_f = t_f.values
                t_f = np.delete(t_f, 0, 1)
                self.train_f = np.append(self.train_f, t_f)
                self.train_label = np.append(self.train_label, i)
        self.train_f = np.reshape(self.train_f, [len(self.train_name), -1])
        print('\nTraining data\n')
        print(self.train_f.shape)
        print(self.train_label.shape)
        self.train_label = self.train_label.astype(int)
        self.train_f = min_max(self.train_f, axis=1)

        test_path = sorted(glob.glob('C:\\Users\\shira\\data_folder\\test_data\\cotton\\Trans_*.csv'))
        for k in range(0, len(test_path)):
            te_name = os.path.basename(test_path[k])
            for l, name in enumerate(self.target, 0):
                if name in te_name:
                    self.test_label = np.append(self.test_label, l)
                else:
                    pass
            te_f = pd.read_csv(test_path[k], header=None, names=["frequency", "transmittance"])
            te_f = te_f.query(frequency_range)
            te_f = te_f.values
            te_f = np.delete(te_f, 0, 1)
            self.test_f = np.append(self.test_f, te_f)
            self.test_name = np.append(self.test_name, te_name)
        self.test_f = np.reshape(self.test_f, [len(test_path), -1])
        self.test_label = np.array(self.test_label)
        self.test_label = self.test_label.astype(int)
        print('\nTest data\n')
        print(self.test_f.shape)
        print(self.test_label.shape)
        self.test_f = min_max(self.test_f, axis=1)

    def svm(self):

        global best_method, gamma, C, switch
        print('\nSVM ( Support Vector Machine )\n')
        param_list = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000]
        best_score = 0
        best_parameters = {}
        kernel = 'rbf'
        result = np.empty([len(self.test_name)], dtype=object)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        for gamma in param_list:
            for C in param_list:
                estimator = SVC(gamma=gamma, kernel=kernel, C=C)
                one_vs_rest_classifier = OneVsRestClassifier(estimator)
                scores = cross_val_score(one_vs_rest_classifier, self.train_f, self.train_label, cv=skf)
                one_vs_rest_score = np.mean(scores)
                one_vs_one_classifier = SVC(gamma=gamma, kernel=kernel, C=C)
                scores2 = cross_val_score(one_vs_one_classifier, self.train_f, self.train_label, cv=skf)
                one_vs_one_score = np.mean(scores2)
                if one_vs_rest_score > one_vs_one_score:
                    score = one_vs_rest_score
                    method = 'One-versus-the-rest'
                    switch = 1
                else:
                    score = one_vs_one_score
                    method = 'One-versus-one'
                    switch = 2
                if score > best_score:
                    best_method = method
                    best_score = score
                    best_parameters = {'gamma': gamma, 'C': C}

        print('Best score: {}'.format(best_score))
        print('Best parameters: {}'.format(best_parameters))
        print('Best method: {}\n'.format(best_method))

        number_of_target = np.zeros(len(self.target))
        for i in range(0, len(self.test_name)):
            for j, name in enumerate(self.target, 0):
                if name in self.test_name[i]:
                    number_of_target[j] += 1
        correct = np.zeros(len(self.target))

        if switch == 1:
            estimator = SVC(**best_parameters, kernel=kernel)
            classifier = OneVsRestClassifier(estimator)
            classifier.fit(self.train_f, self.train_label)
            prediction = classifier.predict(self.test_f)
            for k in range(0, len(self.test_name)):
                print(prediction)
                pred = prediction[k]
                for l, name in enumerate(self.target, 0):
                    if pred == l:
                        result[k] = name
                        if name in self.test_name[k]:
                            correct[l] += 1
        else:
            classifier = SVC(**best_parameters, kernel=kernel)
            classifier.fit(self.train_f, self.train_label)
            prediction = classifier.predict(self.test_f)
            for k in range(0, len(self.test_name)):
                print(prediction)
                pred = prediction[k]
                for l, name in enumerate(self.target, 0):
                    if pred == l:
                        result[k] = name
                        if name in self.test_name[k]:
                            correct[l] += 1

        df_result = pd.DataFrame(result, index=self.test_name.T, columns=['precision'])
        accuracy_class = correct / number_of_target
        accuracy = np.average(accuracy_class)
        accuracy = np.append(accuracy_class, accuracy)
        self.target = np.array(self.target)
        index = np.append(self.target, 'all')
        df_accuracy = pd.DataFrame(accuracy, index=index, columns=['accuracy'])
        print('\nPrecision result\n')
        print(df_result)
        print('\nAccuracy\n')
        print(df_accuracy)

    def random_forest(self):

        print('\nRandom Forest\n')
        param_grid = {'n_estimators': [i for i in range(100, 200, 20)],
                      'criterion': ['gini', 'entropy'],
                      'max_depth': [5, 10, 15],
                      'max_features': [30, 40, 50]}
        result = np.empty([len(self.test_name)], dtype=object)

        rf = RandomForestClassifier(random_state=0, bootstrap=True, n_jobs=-1)
        grid_search = GridSearchCV(rf, param_grid, cv=5)

        grid_search.fit(self.train_f, self.train_label)

        print('Best validation score: {}'.format(grid_search.best_score_))
        print('Best parameters: {}'.format(grid_search.best_params_))

        number_of_target = np.zeros(len(self.target))
        for i in range(0, len(self.test_name)):
            for j, name in enumerate(self.target, 0):
                if name in self.test_name[i]:
                    number_of_target[j] += 1
        correct = np.zeros(len(self.target))

        best = grid_search.best_estimator_
        prediction = best.predict(self.test_f)
        for k in range(0, len(self.test_name)):
            print(prediction)
            pred = prediction[k]
            for l, name in enumerate(self.target, 0):
                if pred == l:
                    result[k] = name
                    if name in self.test_name[k]:
                        correct[l] += 1

        df_result = pd.DataFrame(result, index=self.test_name.T, columns=['precision'])
        accuracy_class = correct / number_of_target
        accuracy = np.average(accuracy_class)
        accuracy = np.append(accuracy_class, accuracy)
        self.target = np.array(self.target)
        index = np.append(self.target, 'all')
        df_accuracy = pd.DataFrame(accuracy, index=index, columns=['accuracy'])
        print('\nPrecision result\n')
        print(df_result)
        print('\nAccuracy\n')
        print(df_accuracy)

    def knn(self):

        print('\nk-nn( k-nearest neighbor )\n')
        param_grid = {'n_neighbors': [1, 2, 3, 4, 5],
                      'weights': ['uniform', 'distance']}
        result = np.empty([len(self.test_name)], dtype=object)

        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(knn, param_grid, cv=5)

        grid_search.fit(self.train_f, self.train_label)

        print('Best validation score: {}'.format(grid_search.best_score_))
        print('Best parameters: {}'.format(grid_search.best_params_))

        number_of_target = np.zeros(len(self.target))
        for i in range(0, len(self.test_name)):
            for j, name in enumerate(self.target, 0):
                if name in self.test_name[i]:
                    number_of_target[j] += 1
        correct = np.zeros(len(self.target))

        best = grid_search.best_estimator_
        prediction = best.predict(self.test_f)
        for k in range(0, len(self.test_name)):
            print(prediction)
            pred = prediction[k]
            for l, name in enumerate(self.target, 0):
                if pred == l:
                    result[k] = name
                    if name in self.test_name[k]:
                        correct[l] += 1

        df_result = pd.DataFrame(result, index=self.test_name.T, columns=['precision'])
        accuracy_class = correct / number_of_target
        accuracy = np.average(accuracy_class)
        accuracy = np.append(accuracy_class, accuracy)
        self.target = np.array(self.target)
        index = np.append(self.target, 'all')
        df_accuracy = pd.DataFrame(accuracy, index=index, columns=['accuracy'])
        print('\nPrecision result\n')
        print(df_result)
        print('\nAccuracy\n')
        print(df_accuracy)


identification = Identification(1.1, 1.8, ['glu', 'lac', 'mal'])
identification.labeling()
identification.svm()
identification.random_forest()
identification.knn()
