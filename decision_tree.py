import argparse
import numpy as np
from math import log2
from collections import Counter
import csv


class Node:
    '''
    Here is an arbitrary Node class that will form the basis of your decision
    tree. 
    Note:
        - the attributes provided are not exhaustive: you may add and remove
        attributes as needed, and you may allow the Node to take in initial
        arguments as well
        - you may add any methods to the Node class if desired 
    '''
    def __init__(self, training_data, labels, depth, max_depth, prev_vote=None, prev_val=None):
        self.attrs = training_data
        self.labels = labels
        self.depth = depth
        self.max_depth = max_depth
        self.children = []
        self.vote = None
        self.prev_vote = "root" if prev_vote == None else prev_vote
        self.prev_val = "root" if prev_val == None else prev_val

    def calculate_label_entropy(self):
        entropy = 0
        counters = Counter(self.labels)
        values = counters.values()
        for value in values:
            entropy += -((value)/sum(values))*log2((value)/sum(values))
        return entropy

    def calculate_conditional_entropy(self, attr):
        conditional_entropy = 0
        attr_counter = Counter(self.attrs[attr])
        values = attr_counter.values()
        for attr_value in values:
            ratio = attr_value/sum(values)
            sub_labels = []
            for i, value in enumerate(self.attrs):
                if value == attr_value:
                    sub_labels.append(self.labels[i])
            sub_labels_counter = Counter(sub_labels)
            sub_entropy = 0
            for sub in sub_labels_counter.values():
                sub_entropy += -((sub)/sum(sub_labels_counter.values()))*log2((sub)/sum(sub_labels_counter.values()))
            
            conditional_entropy += ratio*sub_entropy

        return conditional_entropy


    def calculate_mutual_information(self, attr):
        hy = self.calculate_label_entropy()
        hyx = self.calculate_conditional_entropy(attr)
        return hy - hyx

    def majority_vote(self, data):
        return Counter(data).most_common(1)[0][0]

    def train(self):
        if self.depth == self.max_depth:
            self.vote = self.majority_vote(self.labels)
            return
        if len(self.attrs) == 0:
            self.vote = self.majority_vote(self.labels)
            return 
        
        label_entropy = self.calculate_label_entropy()
        if label_entropy == 0:
            return
        
        mutual_informations = {}
        for attr in self.attrs:
            mutual_informations[attr] = self.calculate_mutual_information(attr)
        max_mi_attr = max(mutual_informations, key=mutual_informations.get)
        self.vote = max_mi_attr

        for value in list(set(self.attrs[max_mi_attr])):
            indexes = [i for i, v in enumerate(self.attrs[max_mi_attr]) if v == value]
            sub_data = {}

            for key in self.attrs:
                if key != max_mi_attr:
                    sub_data[key] = [self.attrs[key][i] for i in indexes]

            sub_labels = [self.labels[i] for i in indexes]
            self.children.append(Node(sub_data, sub_labels, self.depth+1, self.max_depth, self.vote, value))
        
        for child in self.children:
            child.train()
        
        return

    def predict(self, attributes):
        if len(self.attrs) == 0 or self.depth == self.max_depth:
            return self.vote
        val = attributes[self.vote]
        values = list(set(self.attrs[self.vote]))
        for i, value in enumerate(values):
            if val == value:
                return self.children[i].predict(attributes)

    def print(self, file):
        print(f"{'|'*self.depth}{Counter(self.labels)}{self.prev_vote}={self.prev_val} --> {self.vote}", file=file)
        for child in self.children:
            child.print(file)
        return
    

class DecisionTree:
    
    def __init__(self, training_file, max_depth):
        self.training_data, self.labels = self.parse_data(training_file)
        self.root = Node(self.training_data, self.labels, 0, max_depth)

    def parse_data(self, file):
        with open(file, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            data = np.array([row for row in reader])
        
        labels = data[1:,-1]
        dataMap = {title:[] for title in data[0][:-1]}
        for i, title in enumerate(data[0][:-1]):
            dataMap[title] = data[1:,i]
        return dataMap, labels


    def train(self):
        self.root.train()

    def predict(self, attrs):
        predictions = []
        for i in range(len(attrs[list(attrs.keys())[0]])):
            pred = self.root.predict({key: value[i] for key, value in attrs.items()})
            predictions.append(pred)
        return predictions

    def print_trees(self, file=None):
        self.root.print(file)
    
    def print_dataMap(self):
        print(self.training_data)


def calculate_error(labels, predictions):
    error = 0
    for i, label in enumerate(labels):
        if label != predictions[i]:
            error += 1
    return error/len(labels)


if __name__ == '__main__':
    # This takes care of command line argument parsing for you!
    # To access a specific argument, simply access args.<argument name>.
    # For example, to get the train_input path, you can use `args.train_input`.
    parser = argparse.ArgumentParser()
    parser.add_argument("train_input", type=str, help='path to training input .tsv file')
    parser.add_argument("test_input", type=str, help='path to the test input .tsv file')
    parser.add_argument("max_depth", type=int, 
                        help='maximum depth to which the tree should be built')
    parser.add_argument("train_out", type=str, 
                        help='path to output .txt file to which the feature extractions on the training data should be written')
    parser.add_argument("test_out", type=str, 
                        help='path to output .txt file to which the feature extractions on the test data should be written')
    parser.add_argument("metrics_out", type=str, 
                        help='path of the output .txt file to which metrics such as train and test error should be written')
    parser.add_argument("print_out", type=str,
                        help='path of the output .txt file to which the printed tree should be written')
    args = parser.parse_args()
    
    #Here's an example of how to use argparse
    train_input = args.train_input
    test_input = args.test_input
    max_depth = args.max_depth
    train_out = args.train_out
    test_out = args.test_out
    metrics_out = args.metrics_out
    print_out = args.print_out

    tree = DecisionTree(train_input, max_depth)
    tree.train()
    tree.print_trees()

    attrs, labels = tree.parse_data(train_input)
    with open(train_out, "w") as file:
        pred = tree.predict(attrs)
        train_error = calculate_error(labels, pred)
        file.write(f"{pred}\n")

    
    attrs_test, labels_test = tree.parse_data(test_input)
    with open(test_out, "w") as file:
        pred_test = tree.predict(attrs_test)
        test_error = calculate_error(labels_test, pred_test)
        file.write(f"{pred_test}\n")

    with open(metrics_out, "w") as file:
        file.write(f"error(train): {train_error}\n")
        file.write(f"error(test): {test_error}\n")

    # Here is a recommended way to print the tree to a file
    with open(print_out, "w") as file:
        tree.print_trees(file)
        # print_tree(dTree, file)