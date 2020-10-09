import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

new_york_tweets = pd.read_json('new_york.json', lines=True)
#print(len(new_york_tweets))
#print(new_york_tweets.columns)
#print(new_york_tweets.loc[12]["text"])

london_tweets = pd.read_json('london.json', lines=True)
paris_tweets = pd.read_json('paris.json', lines=True)
#print(len(london_tweets))
#print(len(paris_tweets))

new_york_text = new_york_tweets['text'].tolist()
paris_text = paris_tweets['text'].tolist()
london_text = london_tweets['text'].tolist()

all_tweets = new_york_text + paris_text + london_text
labels = [0] * len(new_york_text) + [1] * len(london_text) + [2] * len(paris_tweets)

#print(len(new_york_text))

training_data, testing_data, training_labels, testing_labels = train_test_split(all_tweets, labels, random_state = 1, test_size = 0.2)
#print(len(training_data))
#print(len(testing_data))

counter = CountVectorizer()
print(counter.fit(training_data))
training_counts = counter.transform(training_data)
testing_counts = counter.transform(testing_data)
#print(training_data[3])
#print(training_counts[3])

classifier = MultinomialNB()
classifier.fit(training_counts, training_labels)
predictions = classifier.predict(testing_counts)
#print(accuracy_score(testing_labels, predictions))
#print(predictions)
#print(testing_counts)
my_tweet = 'I am indeed entertaining various aspects'
tweet_counts = counter.transform([my_tweet])
print(classifier.predict(tweet_counts))