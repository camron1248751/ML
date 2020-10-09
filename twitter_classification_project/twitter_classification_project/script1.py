import pandas as pd
import numpy as np
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
all_tweets = pd.read_json('random_tweets.json', lines=True)
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


#print(all_tweets.loc[0]['user'])
#print(len(all_tweets))
#print(all_tweets.columns)
#print(all_tweets.loc[0]['text'])
#print(all_tweets.loc[0])

all_tweets['is_viral'] = np.where(all_tweets['retweet_count'] > 5, 1, 0)
all_tweets['tweet_length'] = all_tweets.apply(lambda tweet: len(tweet['text']), axis=1)
all_tweets['followers_count'] = all_tweets.apply(lambda tweet: tweet['user']['followers_count'], axis=1)
all_tweets['friends_count'] = all_tweets.apply(lambda tweet: tweet['user']['friends_count'], axis=1)

labels = all_tweets['is_viral']

data = all_tweets[['tweet_length', 'followers_count', 'friends_count']] 
scaled_data = scale(data, axis = 0)
print(scaled_data[0])
print(data)
training_data, testing_data, training_labels, testing_labels = train_test_split(scaled_data, labels, test_size = 0.2, random_state = 1)
classifier = KNeighborsClassifier(n_neighbors = 5)
classifier.fit(training_data, training_labels)
print(classifier.score(testing_data, testing_labels))


scores = []
for k in range(1, 201):
    classifier = KNeighborsClassifier(n_neighbors = k)
    classifier.fit(training_data, training_labels)
    scores.append(classifier.score(testing_data, testing_labels))

plt.plot(range(1, 201), scores)
plt.xlabel('k')
plt.ylabel('accuracy_score')
plt.show()