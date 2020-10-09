from reviews import neg_list, pos_list
from sklearn.feature_extraction.text import CountVectorizer

#format data, right now the counts arent associated with the labels, which we will do next

review = "This crib was amazing"
counter = CountVectorizer()
counter.fit(neg_list + pos_list)
print(counter.vocabulary_)
review_counts = counter.transform([review])
print(review_counts.toarray())
training_counts = counter.transform(neg_list + pos_list)
print(training_counts)

#associating the counts with labels (labels meaning 0 or 1 for negative or positive), so we can train the data, then test

from reviews import counter, training_counts
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

review = "amazing I love"
review_counts = counter.transform([review])
training_labels = [0] * 1000 + [1] * 1000
classifier = MultinomialNB()

classifier.fit(training_counts, training_labels)
print(classifier.predict(review_counts))
print(classifier.predict_proba(review_counts))