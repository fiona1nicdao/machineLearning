"""
# def most_common(lst):
#     return max(set(lst), key=lst.count)
# def euclidean(point, data):
#     # Euclidean distance between points a & data
#     return np.sqrt(np.sum((point - data)**2, axis=1))
# class KNeighborsClassifier:
#     def __init__(self, k=5, dist_metric=euclidean):
#         self.k = k
#         self.dist_metric = dist_metric
#     def fit(self, X_train, y_train):
#         self.X_train = X_train
#         self.y_train = y_train
#     def predict(self, X_test):
#         # print("hello")
#         neighbors = []
#         for x in X_test:
#             # print(x)
#             # print(self.X_train)
#             distances = self.dist_metric(x, self.X_train)
#             y_sorted = [y for _, y in sorted(zip(distances, self.y_train))]
#             neighbors.append(y_sorted[:self.k])
#         return list(map(most_common, neighbors))
#     def evaluate(self, X_test, y_test):
#         y_pred = self.predict(X_test)
#         accuracy = sum(y_pred == y_test) / len(y_test)
#         # print("hello")
#         return accuracy
"""

# knn = KNeighborsClassifier(k=5)
# knn.fit(X_train, y_train)
# prediction = knn.predict( X_develop)
# accuracy = knn.evaluate(X_develop, y_develop)
# knn = knn.predict(X_develop)
# evaluate = knn.

# lr_0_0001 = LogisticRegression(C=0.0001).fit(X_train, y_train)
# lr_0_001 = LogisticRegression(C=0.001).fit(X_train, y_train)
# lr_0_01 = LogisticRegression(C=0.01).fit(X_train, y_train)
# lr_0_1 = LogisticRegression(C=0.1).fit(X_train, y_train)
# # lr_0 = LogisticRegression(C=0).fit(X_train, y_train)
# lr_1 = LogisticRegression(C=1).fit(X_train, y_train)
# lr_10 = LogisticRegression(C=10).fit(X_train, y_train)
# lr_100 = LogisticRegression(C=100).fit(X_train, y_train)
# print("\n\n part 2 \n\n")
# lr_0_0001_accuracy = lr_0_0001.score(X_develop, y_develop)
# print(lr_0_0001_accuracy)
# lr_0_001_accuracy = lr_0_001.score(X_develop, y_develop)
# print(lr_0_001_accuracy)
# lr_0_01_accuracy = lr_0_01.score(X_develop, y_develop)
# print(lr_0_01_accuracy)
# lr_0_1_accuracy = lr_0_1.score(X_develop, y_develop)
# print(lr_0_1_accuracy)
# # lr_0_accuracy = lr_0.score(X_develop, y_develop)
# # print(lr_0_accuracy)
# lr_1_accuracy = lr_1.score(X_develop, y_develop)
# print(lr_1_accuracy)
# lr_10_accuracy = lr_10.score(X_develop, y_develop)
# print(lr_10_accuracy)
# lr_100_accuracy =lr_100.score(X_develop, y_develop)
# print(lr_100_accuracy)