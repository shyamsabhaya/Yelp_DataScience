#reference: http://nicolas-hug.com/blog/matrix_facto_4
import numpy as np
import math
from sklearn.metrics import mean_squared_error
import random
import pandas as pd
import sys

class SVDApproximation:
    def fit(self, users, businesses, ratings, n_factors = 10, learning_rate = .005, n_epochs = 100):
        self.user_keys = {}
        self.business_keys = {}

        distinct_users = np.unique(users)
        distinct_businesses = np.unique(businesses)

        # print("generating keys")
        key = 0
        for user in distinct_users:
            self.user_keys[user] = key
            key += 1
        key = 0
        for business in distinct_businesses:
            self.business_keys[business] = key
            key += 1

        # make a random initialization of user and item factors
        p = np.random.normal(0, .1, (len(distinct_users), int(n_factors)))
        q = np.random.normal(0, .1, (len(distinct_businesses), int(n_factors)))

        #print("filling in matrix")
        self.useritems_matrix = np.zeros((len(distinct_users), len(distinct_businesses)))
        for i in range(len(users)):
            self.useritems_matrix[self.user_keys[users[i]], self.business_keys[businesses[i]]] = ratings[i]

        self.globalbaseline = np.mean([
            np.mean([
                r for r in v if r != 0
            ])
            for v in self.useritems_matrix
        ])
        self.user_effects = np.array([
            (np.mean([r for r in v if r != 0]) - self.globalbaseline)
            for v in self.useritems_matrix
        ])
        self.business_effects = np.array([
            (np.mean([r for r in self.useritems_matrix[:, b] if r != 0]) - self.globalbaseline)
            for b in range(len(self.useritems_matrix[0]))
        ])

        # correcting for bias
        # for i in range(len(distinct_users)):
        #     for j in range(len(distinct_businesses)):
        #         self.useritems_matrix[i, j] = (self.useritems_matrix[i,j] - \
        #             (self.user_effects[i] + self.business_effects[j] + self.globalbaseline)) if self.useritems_matrix[i, j] != 0 else 0

        adjusted_ratings = np.zeros((len(ratings),))
        for i in range(len(ratings)):
            adjusted_ratings[i] = ratings[i] - (self.user_effects[self.user_keys[users[i]]]
                                                + self.business_effects[self.business_keys[businesses[i]]]
                                                + self.globalbaseline)

        np.seterr(all='raise')

        #print("stochastic gradient descent")
        # Optimization via Stochastic Gradient Descent
        for nil in range(int(n_epochs)):
            for i in range(len(users)):
                try:
                    error = adjusted_ratings[i] - np.dot(p[self.user_keys[users[i]]], q[self.business_keys[businesses[i]]])
                    # Update the vectors p and q
                    lastp = p[self.user_keys[users[i]]]
                    
                    p[self.user_keys[users[i]]] += learning_rate * error * q[self.business_keys[businesses[i]]]
                    q[self.business_keys[businesses[i]]] += learning_rate * error * lastp
                except:
                    print(nil)
                    print(error)
                    print(lastp)
                    print(users[i] + ' ' + businesses[i] + ' ' + str(ratings[i]))
                    exit()

        self.p, self.q = p, q


    def GetUserMean(self, user):
        return np.mean([i for i in self.useritems_matrix[self.user_keys[user]] if i != 0])

    def predict(self, users, businesses):
        predictions = np.zeros((len(users),))
        for i in range(len(users)):

            try:
                bias = self.user_effects[self.user_keys[users[i]]] + \
                       self.business_effects[self.business_keys[businesses[i]]] + \
                       self.globalbaseline
                predictions[i] = \
                    np.dot(self.p[self.user_keys[users[i]]],
                           self.q[self.business_keys[businesses[i]]]) + bias
            except:
                predictions[i] = self.GetUserMean(users[i])

        return predictions

    def evaluate(self, predictions, truth):
        return math.sqrt(mean_squared_error(predictions, truth))

def RandomPartition(lst, n):
    random.shuffle(lst)
    division = len(lst) / float(n)
    return [lst[int(round(division * i)): int(round(division * (i + 1)))] for i in range(n)]


def KFoldsSplit(users, businesses, ratings, k=3):
    user_counts = {}
    user_indices = {}
    for i in range(len(users)):
        if users[i] not in user_counts.keys():
            user_counts[users[i]] = 1
            user_indices[users[i]] = [i]
        else:
            user_counts[users[i]] += 1
            user_indices[users[i]].append(i)

    users_ = [[] for i in range(k)]
    businesses_ = [[] for i in range(k)]
    ratings_ = [[] for i in range(k)]

    commonusers = []
    commonbusinesses = []
    commonratings = []

    for user, count in user_counts.items():
        if count >= k:
            indices = RandomPartition(user_indices[user], k)
            for i in range(k):
                users_[i] += [users[j] for j in indices[i]]
                businesses_[i] += [businesses[j] for j in indices[i]]
                ratings_[i] += [ratings[j] for j in indices[i]]
        else:
            commonusers += [users[j] for j in user_indices[user]]
            commonbusinesses += [businesses[j] for j in user_indices[user]]
            commonratings += [ratings[j] for j in user_indices[user]]

    return users_, businesses_, ratings_, commonusers, commonbusinesses, commonratings


def main1():
    training_data = pd.read_csv(sys.argv[1])
    testing_data = pd.read_csv(sys.argv[2])
    data = training_data[['user_id', 'business_id', 'review_rating']]
    users = data['user_id']
    businesses = data['business_id']
    ratings = data['review_rating']

    distinct_users = np.unique(users)

    rec = SVDApproximation()

    print('Fitting')
    rec.fit(users.values, businesses.values, ratings.values)

    print('Predicting')

    todrop = []
    for index, row in testing_data.iterrows():
        if not (row['user_id'] in distinct_users):
            todrop.append(index)

    testing_data = testing_data.drop(testing_data.index[todrop])
    users = testing_data['user_id']
    businesses = testing_data['business_id']
    target_truth = testing_data['review_rating']
    pred = rec.predict(users.values, businesses.values)
    # print(pred)
    np.savetxt('meansvdresult.txt', pred, fmt='%.2e')

    print('Evaluating')
    print(rec.evaluate(pred, target_truth))


def main2():
    training_data = pd.read_csv(sys.argv[1])
    testing_data = pd.read_csv(sys.argv[2])
    data = training_data[['user_id', 'business_id', 'review_rating']]
    users = data['user_id']
    businesses = data['business_id']
    ratings = data['review_rating']

    distinct_users = np.unique(users)

    rec = SVDApproximation()

    print('splitting')
    user_splits, business_splits, rating_splits, commonusers, commonbusinesses, commonratings = KFoldsSplit(users,
                                                                                                            businesses,
                                                                                                            ratings, 3)
    print('done splitting')

    feature_candidates = np.arange(9,11,2)
    epoch_candidates = [5]
    scores = np.zeros((len(feature_candidates) * len(epoch_candidates), 3))
    factors = np.zeros((len(feature_candidates) * len(epoch_candidates), 3))
    epochs = np.zeros((len(feature_candidates) * len(epoch_candidates), 3))
    for i in range(3):
        if i == 0:
            training_users = commonusers + user_splits[1] + user_splits[2]
            training_businesses = commonbusinesses + business_splits[1] + business_splits[2]
            training_ratings = commonratings + rating_splits[1] + rating_splits[2]
            testing_users = user_splits[0]
            testing_businesses = business_splits[0]
            testing_ratings = rating_splits[0]
        elif i == 2:
            training_users = commonusers + user_splits[0] + user_splits[1]
            training_businesses = commonbusinesses + business_splits[0] + business_splits[1]
            training_ratings = commonratings + rating_splits[0] + rating_splits[1]
            testing_users = user_splits[2]
            testing_businesses = business_splits[2]
            testing_ratings = rating_splits[2]
        else:
            training_users = commonusers + user_splits[0] + user_splits[2]
            training_businesses = commonbusinesses + business_splits[0] + business_splits[2]
            training_ratings = commonratings + rating_splits[0] + rating_splits[2]
            testing_users = user_splits[1]
            testing_businesses = business_splits[1]
            testing_ratings = rating_splits[1]

        print('Fitting split ' + str(i + 1))

        for j in range(len(feature_candidates)):
            print(j)
            for k in range(len(epoch_candidates)):
                rec.fit(training_users, training_businesses, training_ratings, n_factors=feature_candidates[j], n_epochs=epoch_candidates[k])
                scores[((j - 1) * len(epoch_candidates)) + k, i] = rec.evaluate(rec.predict(testing_users, testing_businesses), testing_ratings)
                factors[((j - 1) * len(epoch_candidates)) + k, i] = feature_candidates[j]
                epochs[((j - 1) * len(epoch_candidates)) + k, i] = epoch_candidates[k]

    averages = np.zeros((len(feature_candidates) * len(epoch_candidates),))
    index = -1
    min = -1
    for i in range(len(scores)):
        averages[i] = np.mean(scores[i])
        if index == -1 or min > averages[i]:
            min = averages[i]
            index = i

    print("Best k = " + str(factors[index,0]) + ", epochs:  = " + str(epochs[index,0]) + " at average cv rmse: " + str(min))

    print('Predicting')

    todrop = []
    for i, row in testing_data.iterrows():
        if not (row['user_id'] in distinct_users):
            todrop.append(i)

    print('Fitting')
    rec.fit(users.values, businesses.values, ratings.values, n_factors=factors[index,0],
            n_epochs=epochs[index,0])
    testing_data = testing_data.drop(testing_data.index[todrop])
    users = testing_data['user_id']
    businesses = testing_data['business_id']
    target_truth = testing_data['review_rating']

    pred = rec.predict(users.values, businesses.values)
    # print(pred)
    np.savetxt('meansvdresult.txt', pred, fmt='%.2e')

    print('Evaluating')
    print(rec.evaluate(pred, target_truth))


if __name__ == '__main__':
    main2()
