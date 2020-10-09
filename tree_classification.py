from collections import Counter

cars = [['med', 'low', '3', '4', 'med', 'med'], ['med', 'vhigh', '4', 'more', 'small', 'high'], ['high', 'med', '3', '2', 'med', 'low'], ['med', 'low', '4', '4', 'med', 'low'], ['med', 'low', '5more', '2', 'big', 'med'], ['med', 'med', '2', 'more', 'big', 'high'], ['med', 'med', '2', 'more', 'med', 'med'], ['vhigh', 'vhigh', '2', '2', 'med', 'low'], ['high', 'med', '4', '2', 'big', 'low'], ['low', 'low', '2', '4', 'big', 'med']]

car_labels = ['acc', 'acc', 'unacc', 'unacc', 'unacc', 'vgood', 'acc', 'unacc', 'unacc', 'good']

def split(dataset, labels, column):
    data_subsets = []
    label_subsets = []
    counts = list(set([data[column] for data in dataset]))
    counts.sort()
    for k in counts:
        new_data_subset = []
        new_label_subset = []
        for i in range(len(dataset)):
            if dataset[i][column] == k:
                new_data_subset.append(dataset[i])
                new_label_subset.append(labels[i])
        data_subsets.append(new_data_subset)
        label_subsets.append(new_label_subset)
    return data_subsets, label_subsets

def gini(dataset):
  impurity = 1
  label_counts = Counter(dataset)
  for label in label_counts:
    prob_of_label = label_counts[label] / len(dataset)
    impurity -= prob_of_label ** 2
  return impurity

def information_gain(starting_labels, split_labels):
  info_gain = gini(starting_labels)
  for subset in split_labels:
    info_gain -= gini(subset) * len(subset) / len(starting_labels)
  return info_gain

for i in range(6):
  split_data, split_labels = split(cars, car_labels, i)
  print(information_gain(car_labels, split_labels))

#Making the tree


car_data = [['med', 'low', '3', '4', 'med', 'med'], ['med', 'vhigh', '4', 'more', 'small', 'high'], ['high', 'med', '3', '2', 'med', 'low'], ['med', 'low', '4', '4', 'med', 'low'], ['med', 'low', '5more', '2', 'big', 'med'], ['med', 'med', '2', 'more', 'big', 'high'], ['med', 'med', '2', 'more', 'med', 'med'], ['vhigh', 'vhigh', '2', '2', 'med', 'low'], ['high', 'med', '4', '2', 'big', 'low'], ['low', 'low', '2', '4', 'big', 'med']]

car_labels = ['acc', 'acc', 'unacc', 'unacc', 'unacc', 'vgood', 'acc', 'unacc', 'unacc', 'good']

def find_best_split(dataset, labels):
    best_gain = 0
    best_feature = 0
    for feature in range(len(dataset[0])):
        data_subsets, label_subsets = split(dataset, labels, feature)
        gain = information_gain(labels, label_subsets)
        if gain > best_gain:
            best_gain, best_feature = gain, feature
    return best_feature, best_gain

def build_tree(data, labels):
  best_feature, best_gain = find_best_split(data, labels)
  if best_gain == 0:
    return Counter(labels)
  data_subsets, label_subsets = split(data, labels, best_feature)
  branches = []
  for i in range(len(data_subsets)):
    branches.append(build_tree(data_subsets[i], label_subsets[i]))
  return branches

print(build_tree(car_data, car_labels))
tree = build_tree(car_data, car_labels)

#make da tree

car_labels = ['acc', 'acc', 'unacc', 'unacc', 'unacc', 'vgood', 'acc', 'unacc', 'unacc', 'good']

def find_best_split(dataset, labels):
    best_gain = 0
    best_feature = 0
    for feature in range(len(dataset[0])):
        data_subsets, label_subsets = split(dataset, labels, feature)

Output:
[Counter({'unacc': 4}), [Counter({'good': 1}), Counter({'acc': 1}), Counter({'unacc': 1})], [Counter({'vgood': 1}), Counter({'acc': 1}), Counter({'acc': 1})]]
#aka
Splitting
--> Branch 0:
  Counter({'unacc': 4})
--> Branch 1:
  Splitting
  --> Branch 0:
    Counter({'good': 1})
  --> Branch 1:
    Counter({'acc': 1})
  --> Branch 2:
    Counter({'unacc': 1})
--> Branch 2:
  Splitting
  --> Branch 0:
    Counter({'vgood': 1})
  --> Branch 1:
    Counter({'acc': 1})
  --> Branch 2:
    Counter({'acc': 1})

#for above, theres just no label on what feature is being used to split, but the first split is safety,
#which is why unacc is immediately shown
#once you have the training_data tree, use it to classify new points:

from tree import *
import operator

test_point = ['vhigh', 'low', '3', '4', 'med', 'med']
#print_tree(tree)
def classify(datapoint, tree):
  if isinstance(tree, Leaf):
    return max(tree.labels.items(), key=operator.itemgetter(1))[0]
  value = datapoint[tree.feature]
  for branch in tree.branches:
    if branch.value == value:
      return classify(datapoint, branch)

print(classify(test_point, tree))

#make multiple trees as a forest by splitting the training data. bagging the features for each tree in the forest helps too
classifier = RandomForestClassifier(n_estimators = number_of_trees)