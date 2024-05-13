# DecisionTree_ID3

## Introduction
Basic implementation of the ID3 (mutual information or information gain) decision tree algorithm from scratch.  

## Example
- The following command runs the program on the small dataset and learn a tree with a max_depth of 2. The train and test predictions would be written to small_2_train.txt and small_2_test.txt.
``` python
>>> python decision_tree.py ./handout/small_train.tsv ./handout/small_test.tsv 2        
./small_2_train.txt ./small_2_test.txt ./small_2_metrics.txt ./small_2_print.txt

>>>
Counter({'1': 14, '0': 14})root=root --> chest_pain
|Counter({'1': 12, '0': 4})chest_pain=0 --> thalassemia
||Counter({'1': 4, '0': 3})thalassemia=0 --> 1
||Counter({'1': 8, '0': 1})thalassemia=1 --> 1
|Counter({'0': 10, '1': 2})chest_pain=1 --> thalassemia
||Counter({'0': 7})thalassemia=0 --> 0
||Counter({'0': 3, '1': 2})thalassemia=1 --> 0
```
- The train and test error (unrounded) would be wriiten to small_2_metrics.txt
```
error(train): 0.21428571428571427
error(test): 0.2857142857142857
```

## Reference
This work is inspired by CMU 10601 Intro to Machine Learning. The course material and dataset are open-sourced on the official website.
