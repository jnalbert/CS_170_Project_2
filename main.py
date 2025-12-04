import numpy as np
from search_algos import forward_selection
from search_algos import backward_elimination

# Group: Justin Albert - jalbe020 - Sec 021, Vinden Drummond - vdrum002 - Sec 021, Sarbesh Sankar - ssank019 - Sec 021
# - DatasetID: 123 (I'm not sure what this is for)
# - Small Dataset Results:
# -     Forward: Feature Subset: {3,5}, Acc: 92%
# -     Backward: Feature Subset: {2,4,5,7,1}, Acc: 83%
# - Large Dataset Results:
# -     Forward: Feature Subset: {1,27}, Acc: 95.5%
# -     Backward: Feature Subset: {1,2,3,5,9,10,11,12,13,15,17,18,19,20,22,23,24,25,26,27,28,29,30,31,34,35,36,37,38,40}, Acc: 74.5%
# - Titanic Dataset Results:
# -     Forward: Feature Subset: {2}, Acc: 78%
# -     Backward: Feature Subset: {2}, Acc: 78%


# adapted from https://www.geeksforgeeks.org/python/how-to-normalize-an-numpy-array-so-the-values-range-exactly-between-0-and-1/
def normalize_data(dataset):
    print("Please wait while I normalize the data... ")

    class_labels = []
    # 2 D array of features
    dataset_features = []
    for item in dataset:
        class_labels.append(item[0])
        dataset_features.append(item[1])

    dataset_features = np.array(dataset_features)

    feature_mins = np.min(dataset_features)
    feature_maxs = np.max(dataset_features)

    # normalize the features
    normalized_features = (dataset_features - feature_mins) / (feature_maxs - feature_mins)

    # convert datae back to the data set format of (class_label, feature_vector)
    normalized_dataset = []
    for i in range(len(dataset)):
        normalized_dataset.append((class_labels[i], normalized_features[i].tolist()))

    return normalized_dataset


# adapted from https://stackoverflow.com/questions/29307532/python-how-can-read-as-float-numbers-a-series-of-strings-from-a-text-file
def load_dataset(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()
        dataset = []
        for line in lines:
            line = line.strip()
            if line:
                features = []
                for x in line.split():
                    features.append(float(x))
                dataset.append((features[0], features[1:]))
    return dataset


def main():
    print(
        "Welcome to Justin Albert, Vinden Drummond, and Sarbesh Sankar's Feature Selection Algorithm.\n"
    )

    filename = input("Type in the name of the file to test: ")

    dataset = load_dataset(filename)

    num_features = len(dataset[0][1])
    num_instances = len(dataset)

    print(
        f"This dataset has {num_features} features (not including the class attribute), with {num_instances} instances.\n"
    )

    dataset = normalize_data(dataset)

    print("\nType the number of the algorithm you want to run.\n")
    print("\t1) Forward Selection")
    print("\t2) Backward Elimination")
    print()

    algo_choice = int(input())

    # make a 1 indexed list of features
    features = list(range(1, num_features + 1))

    if algo_choice == 1:
        forward_selection(features, dataset)
    elif algo_choice == 2:
        backward_elimination(features, dataset)


main()
