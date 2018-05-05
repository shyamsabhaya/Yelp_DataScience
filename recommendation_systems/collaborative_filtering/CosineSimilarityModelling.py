# Reference: https://cambridgespark.com/content/tutorials/implementing-your-own-recommender-systems-in-Python/index.html

import pandas as pd
import numpy as np
import sys
import math

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error

class CosSimWeightedBipartiteGraphProjRecommender:
    def __init__(self, n_jobs=1):
        self.n_jobs = n_jobs

    def fit(self, users, businesses, ratings):
        print("Entered Fit")
        self.user_keys = {}
        self.business_keys = {}
        
        distinct_users = np.unique(users)
        distinct_businesses = np.unique(businesses)
        
        self.useritems_matrix = np.zeros((len(distinct_users), len(distinct_businesses)))


        # fiveperc = len(distinct_users) / 20
        # j = 0
        # rating_mean = np.zeros(len(users))
        # for user in distinct_users:
        #     j += 1
        #     if ((j % fiveperc) == 0):
        #         print(str(j * 100 / len(distinct_users)) + " percent done with users")
        #     reviews = []
        #     places = []
        #     for i in range(len(users)):
        #         if user == users[i]:
        #             reviews.append(ratings[i])
        #             places.append(i)
        #     mean = np.mean(reviews)
        #     rating_mean[np.array(places)] = mean
        #
        # print("making ratings adjusted")
        # ratings_adjusted = ratings - rating_mean

        print("generating keys")
        key = 0
        for user in distinct_users:
            self.user_keys[user] = key
            key += 1
        key = 0
        for business in distinct_businesses:
            self.business_keys[business] = key
            key += 1

        print("filling in graph")
        for i in range(len(users)):
            self.useritems_matrix[self.user_keys[users[i]], self.business_keys[businesses[i]]] = ratings[i]

    def predict(self, users, businesses):
        # Retrieve all users that have reviewed a particular business
        businessuser_map = {}
        for business in businesses:
            if business not in businessuser_map.keys():
                try:
                    bus_col = self.useritems_matrix[:, self.business_keys[business]]
                except:
                    bus_col = []
                businessuser_map[business] = []
                for i in range(len(bus_col)):
                    if bus_col[i] != 0:
                        businessuser_map[business].append(i)

        # Make Predictions
        predictions = []
        for i in range(len(users)):
            print("On " + str(i) + " out of " + str(len(users)) + " users")
            #  calculate similarity
            user_data_pool = np.array([self.useritems_matrix[index] for index in businessuser_map[businesses[i]]])
            me = len(user_data_pool)
            try:
                similarities = pairwise_distances(np.vstack((user_data_pool, self.useritems_matrix[self.user_keys[users[i]]])), metric='cosine')
            except:
                similarities = np.array([[0]])  # we are the first to go to this business...

            user_mean = np.mean([i for i in self.useritems_matrix[self.user_keys[users[i]]] if i != 0])

            similarities = np.delete(similarities[me], me)
            try:
                differences = np.hstack((user_data_pool[:, self.business_keys[businesses[i]]]))
                means = np.zeros((len(businessuser_map[businesses[i]])))
                for index in range(len(businessuser_map[businesses[i]])):
                    nonzero = [r for r in self.useritems_matrix[businessuser_map[businesses[i]][index]] if r != 0]
                    means[index] = np.mean(nonzero)
                differences = differences - means
            except:
                differences = np.array([0])

            norm_factor = np.abs(similarities).sum(axis=0)
            # print(norm_factor)
            # print(norm_factor.shape)
            if norm_factor != 0:
                predictions.append(user_mean + similarities.dot(differences) / norm_factor)
            else:
                predictions.append(user_mean)

        return np.array(predictions)

    def recommend(self, user, num = 5):
        my_index = self.user_keys[user]
        business_edges = []
        for i in range(len(self.useritems_matrix[my_index])):
            if self.useritems_matrix[my_index, i] != 0:
                business_edges.append(i)

        reachable_users = []
        for b in business_edges:
            for i in range(len(self.useritems_matrix[:, b])):
                if self.useritems_matrix[i, b] != 0:
                    reachable_users.append(i)

        reachable_users = np.unique(reachable_users)

        transition_probabilities = pairwise_distances(np.vstack((np.array([self.useritems_matrix[i] for i in reachable_users]),
                                        self.useritems_matrix[my_index])),
                                        metric='cosine')
        transition_probabilities = np.delete(transition_probabilities[len(reachable_users)], len(reachable_users))

        # use a normalization factor to make transition probabilities to sum to one
        norm_factor = np.abs(transition_probabilities).sum(axis=0)
        transition_probabilities = transition_probabilities / norm_factor

        recommendations = []
        for i in range(num):
            # randomly walk to a related user
            destination = np.random.choice(reachable_users, p = transition_probabilities)
            their_restaurants = []
            for j in range(len(self.useritems_matrix[destination])):
                if self.useritems_matrix[destination, j] != 0:
                    their_restaurants.append(j)

            # translate business indices to their key - unfortunately no better way
            # to do it than this expensive loop
            keys = []
            for index in their_restaurants:
                for k, v in self.business_keys.items():
                    if v == index:
                        keys.append(k)
                        break

            pred = self.predict([user for i in range(len(keys))], keys)

            # add a business to recommendations
            max = 0
            for j in range(len(pred)):
                if pred[j] > max:
                    best_business = keys[j]

            recommendations.append(best_business)

        return recommendations

    def evaluate(self, predictions, truth):
        return math.sqrt(mean_squared_error(predictions, truth))

def main():
    training_data = pd.read_csv(sys.argv[1])
    testing_data = pd.read_csv(sys.argv[2])
    data = training_data[['user_id', 'business_id', 'review_rating']]
    users = data['user_id']
    businesses = data['business_id']
    ratings = data['review_rating']

    distinct_users = np.unique(users)

    rec = CosSimWeightedBipartiteGraphProjRecommender()

    print('Fitting')
    rec.fit(users.values, businesses.values, ratings.values)

    print('Predicting')
    
    todrop = []
    for index, row in testing_data.iterrows():
        if not( row['user_id'] in distinct_users ):
            todrop.append(index)
            
    testing_data = testing_data.drop(testing_data.index[todrop])
    users = testing_data['user_id']
    businesses = testing_data['business_id']
    target_truth = testing_data['review_rating']
    pred = rec.predict(users.values, businesses.values)
    print(pred)

    np.savetxt('cossimresult.txt', pred, fmt='%.2e')

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

    rec = CosSimWeightedBipartiteGraphProjRecommender()

    print('Fitting')
    rec.fit(users.values, businesses.values, ratings.values)

    print('Recommending ' + distinct_users[5])
    print(rec.recommend(distinct_users[5]))

if __name__ == '__main__':
    main2()

