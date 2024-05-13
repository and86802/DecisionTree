import csv
import math
from math import log2
import sys
import numpy as np
from collections import defaultdict, Counter

def inspect(input_file):
    with open(input_file, 'r') as file:
        reader = csv.reader(file, delimiter='\t')
        data = np.array([row for row in reader])
    
    labels = data[1:, -1]
    entropy = calculate_label_entropy(Counter(labels))
    error = calculate_majority_vote_error(Counter(labels))
    return entropy, error
                

def calculate_label_entropy(counters):
    entropy = 0
    values = counters.values()
    for value in values:
        entropy += -((value)/sum(values))*log2((value)/sum(values))
    return entropy

def calculate_majority_vote_error(counters):
    values = counters.values()
    return 1 - max(values)/sum(values)


def main():
    if len(sys.argv) != 3:
        print("Usage: python inspection.py <input_file> <output_file>")
        return
    
    input_file, output_file = sys.argv[1], sys.argv[2]
    entropy, error = inspect(input_file)
    output_file = open(output_file, 'w')
    output_file.write(f"entropy: {entropy}\n")
    output_file.write(f"error: {error}\n")
    output_file.close()

if __name__ == '__main__':
    main()