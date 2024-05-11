#Dataset Import of Q1:

from sklearn.datasets import load_iris
from sklearn.datasets import make_moons


#loading Datasets:

iris = load_iris()
iris_data = iris.data
iris_target = iris.target

moon_data, moon_target = make_moons(n_samples=200, noise=0.1)

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

#Data plot:

plt.scatter(iris_data[:, 0], iris_data[:, 1], c=iris_target)
plt.show()

plt.scatter(moon_data[:, 0], moon_data[:, 1], c=moon_target)
plt.show()

#Data to DataFrame:

iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target


moon_df = pd.DataFrame(moon_data)
moon_df['target'] = moon_target


#now lets split our data into train and test data , the user can change the amount of split by changing the value (size)

size_iris = float(input("split rate for iris (between 0 and 1)"))
xi_train_val, xi_test, yi_train_val, yi_test = train_test_split(iris_data, iris_target, test_size=size_iris, random_state=42)
xi_train, xi_val, yi_train, yi_val = train_test_split(xi_train_val, yi_train_val, test_size=0.25, random_state=42)

size_moon = float(input("split rate for two moons (between 0 and 1)"))
xm_train_val, xm_test, ym_train_val, ym_test = train_test_split(moon_data, moon_target, test_size=size_moon, random_state=42)
xm_train, xm_val, ym_train, ym_val = train_test_split(xm_train_val, ym_train_val, test_size=0.25, random_state=42)


d = int(input("Dataset: \n1.iris\n2.two moons\n"))
a = int(input("Classifier: \n1.KNN\n2.SVM\n3.Decision Tree\n4.Naive Bayes\n"))

if d == 1:
    if a == 1:
        knn = KNeighborsClassifier(n_neighbors=3)

        knn.fit(xi_train, yi_train)

        yi_pred = knn.predict(xi_test)
        yi_pred == yi_test

        print(classification_report(yi_test, yi_pred))

        confusion_matrix(yi_test, yi_pred)

        plt.scatter(xi_test[:, 0], xi_test[:, 1], c=yi_pred)
        plt.title('Iris knn')
        plt.show()

        k_values = range(1, 30, 2)

        train_accuracies = []
        val_accuracies = []

        for k in k_values:
            knn_model = KNeighborsClassifier(n_neighbors=k)

            knn_model.fit(xi_train, yi_train)

            yi_train_pred = knn_model.predict(xi_train)

            yi_val_pred = knn_model.predict(xi_val)

            train_accuracy = accuracy_score(yi_train, yi_train_pred)
            train_accuracies.append(train_accuracy)

            val_accuracy = accuracy_score(yi_val, yi_val_pred)
            val_accuracies.append(val_accuracy)

        plt.plot(k_values, train_accuracies, label='Training Accuracy')
        plt.plot(k_values, val_accuracies, label='Validation Accuracy')
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('KNN Accuracy vs. Number of Neighbors')
        plt.show()

        for k, train_acc, val_acc in zip(k_values, train_accuracies, val_accuracies):
            print(f'k={k}: Training Accuracy = {train_acc:.4f}, Validation Accuracy = {val_acc:.4f}')

    elif a == 2:
        model = svm.SVC(kernel='linear')

        model.fit(xi_train, yi_train)

        yi_val_pred = model.predict(xi_val)
        print("Accuracy:", accuracy_score(yi_val, yi_val_pred))

        print(classification_report(yi_val, yi_val_pred))

        confusion_matrix(yi_val, yi_val_pred)

        plt.scatter(xi_val[:, 0], xi_val[:, 1], c=yi_val_pred)
        plt.title('iris linear svm')
        plt.show()

        model = svm.SVC(kernel='poly')
        model.fit(xi_train, yi_train)

        yi_val_pred = model.predict(xi_val)
        print("Accuracy:", accuracy_score(yi_val, yi_val_pred))

        print(classification_report(yi_val, yi_val_pred))

        confusion_matrix(yi_val, yi_val_pred)

        plt.scatter(xi_val[:, 0], xi_val[:, 1], c=yi_val_pred)
        plt.title('iris poly svm')
        plt.show()

        model = svm.SVC(kernel='rbf')
        model.fit(xi_train, yi_train)

        yi_val_pred = model.predict(xi_val)
        print("Accuracy:", accuracy_score(yi_val, yi_val_pred))

        print(classification_report(yi_val, yi_val_pred))

        confusion_matrix(yi_val, yi_val_pred)

        plt.scatter(xi_val[:, 0], xi_val[:, 1], c=yi_val_pred)
        plt.title('iris rbf svm')
        plt.show()

    elif a ==3:
        depth_values = [3, 5, 10]

        for depth in depth_values:
            model = DecisionTreeClassifier(max_depth=depth)

            model.fit(xi_train, yi_train)
            yi_val_pred = model.predict(xi_val)
            print("Iris Decision Tree")

            print(f"Accuracy for depth {depth}: ", accuracy_score(yi_val, yi_val_pred))
            print("Classification Report:")
            print(classification_report(yi_val, yi_val_pred))
            print("Confusion Matrix:")
            print(confusion_matrix(yi_val, yi_val_pred))

            plt.figure()
            plot_tree(model, filled=True)
            plt.show()

    elif a == 4:
        model = GaussianNB()
        model.fit(xi_train, yi_train)
        yi_val_pred = model.predict(xi_val)

        print("Accuracy:", accuracy_score(yi_val, yi_val_pred))
        print(classification_report(yi_val, yi_val_pred))
        print(confusion_matrix(yi_val, yi_val_pred))

        plt.scatter(xi_test[:, 0], xi_test[:, 1], c=yi_val_pred)
        plt.title('iris GaussianNB')
        plt.show()


    else:
        print('invalid input')

elif d == 2:

    if a ==1:
        knn = KNeighborsClassifier(n_neighbors=2)

        knn.fit(xm_train, ym_train)

        ym_pred = knn.predict(xm_test)
        ym_pred == ym_test

        print(classification_report(ym_test, ym_pred))

        confusion_matrix(ym_test, ym_pred)

        plt.scatter(xm_test[:, 0], xm_test[:, 1], c=ym_pred)
        plt.title('two moons KNN')
        plt.show()

        k_values = range(1, 50, 2)

        train_accuracies = []
        val_accuracies = []

        for k in k_values:
            knn_model = KNeighborsClassifier(n_neighbors=k)

            knn_model.fit(xm_train, ym_train)

            ym_train_pred = knn_model.predict(xm_train)

            ym_val_pred = knn_model.predict(xm_val)

            train_accuracy = accuracy_score(ym_train, ym_train_pred)
            train_accuracies.append(train_accuracy)

            val_accuracy = accuracy_score(ym_val, ym_val_pred)
            val_accuracies.append(val_accuracy)

        plt.plot(k_values, train_accuracies, label='Training Accuracy')
        plt.plot(k_values, val_accuracies, label='Validation Accuracy')
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('KNN Accuracy vs. Number of Neighbors')
        plt.show()

        for k, train_acc, val_acc in zip(k_values, train_accuracies, val_accuracies):
            print(f'k={k}: Training Accuracy = {train_acc:.4f}, Validation Accuracy = {val_acc:.4f}')

    elif a == 2:
        model = svm.SVC(kernel='linear')
        model.fit(xm_train, ym_train)

        ym_val_pred = model.predict(xm_val)
        print("Accuracy:", accuracy_score(ym_val, ym_val_pred))

        print(classification_report(ym_val, ym_val_pred))

        confusion_matrix(ym_val, ym_val_pred)

        plt.scatter(xm_val[:, 0], xm_val[:, 1], c=ym_val_pred)
        plt.title('two moons linear SVM')
        plt.show()

        model = svm.SVC(kernel='poly')
        model.fit(xm_train, ym_train)

        ym_val_pred = model.predict(xm_val)
        print("Accuracy:", accuracy_score(ym_val, ym_val_pred))

        print(classification_report(ym_val, ym_val_pred))

        confusion_matrix(ym_val, ym_val_pred)

        plt.scatter(xm_val[:, 0], xm_val[:, 1], c=ym_val_pred)
        plt.title('two moons poly SVM')
        plt.show()

        model = svm.SVC(kernel='rbf')
        model.fit(xm_train, ym_train)

        ym_val_pred = model.predict(xm_val)
        print("Accuracy:", accuracy_score(ym_val, ym_val_pred))

        print(classification_report(ym_val, ym_val_pred))

        confusion_matrix(ym_val, ym_val_pred)

        plt.scatter(xm_val[:, 0], xm_val[:, 1], c=ym_val_pred)
        plt.title('two moons rbf SVM')
        plt.show()

    elif a == 3:

        depth_values = [3, 5, 10]

        for depth in depth_values:
            model = DecisionTreeClassifier(max_depth=depth)

            model.fit(xm_train, ym_train)
            ym_val_pred = model.predict(xm_val)
            print("two moons decision tree0")

            print(f"Accuracy for depth {depth}: ", accuracy_score(ym_val, ym_val_pred))
            print("Classification Report:")
            print(classification_report(ym_val, ym_val_pred))
            print("Confusion Matrix:")
            print(confusion_matrix(ym_val, ym_val_pred))

            plt.figure()
            plot_tree(model, filled=True)
            plt.show()

    elif a == 4:
        model = GaussianNB()
        model.fit(xm_train, ym_train)
        ym_val_pred = model.predict(xm_val)

        print("Accuracy:", accuracy_score(ym_val, ym_val_pred))
        print(classification_report(ym_val, ym_val_pred))
        print(confusion_matrix(ym_val, ym_val_pred))

        #plt.scatter(xm_test[:, 0], xm_test[:, 1], c=ym_val_pred)
        #plt.title('two moons GaussianNB')
        #plt.show()