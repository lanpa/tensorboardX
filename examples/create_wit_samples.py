
"""
https://archive.ics.uci.edu/ml/datasets/Heart+Disease
1. #3 (age)
2. #4 (sex)
3. #9 (cp)
4. #10 (trestbps)
5. #12 (chol)
6. #16 (fbs)
7. #19 (restecg)
8. #32 (thalach)
9. #38 (exang)
10. #40 (oldpeak)
11. #41 (slope)
12. #44 (ca)
13. #51 (thal)
14. #58 (num) (the predicted attribute)

ex: ['62', '0', '1', '140', '0', '?', '0', '143', '0', '0', '?', '?', '3', '2']

Here I only choose feature 1~8 for example.

"""

import tensorflow as tf

with open("processed.cleveland.data") as f:
    lines = f.readlines()

columns = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "target"]
features_targets = []
targets = []
for line in lines:
    x = line.strip().split(',')
    y = x[-1]
    if '?' in x[:8]:  # skip missing data
        continue
    ft = [float(i) for i in x[:8]+[y]]
    print(ft)
    features_targets.append(ft)

# print(columns)
# print(features_targets)

def to_examples(features_targets, columns=None):
    examples = []
    for row in features_targets:
        example = tf.train.Example()
        for i, col in enumerate(columns):
            example.features.feature[col].float_list.value.append(row[i])
            # example.features.feature[col].bytes_list.value.append(row[col].encode('utf-8'))
        examples.append(example)
    return examples

writer = tf.io.TFRecordWriter('test.tfrecord')
for example in to_examples(features_targets, columns):
    writer.write(example.SerializeToString())
writer.close()


# fill [git/]tensorboardX/examples/test.tfrecord in the WIT page and see the data distribution.
