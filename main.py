import time
from search_algos import forward_selection
from search_algos import backward_elimination
from validator import Validator
from nn_classifier import NearestNeighborClassifier
# def main():
  
#   print("Welcome to Justin Albert, Vinden Drummond, and Sarbesh Sankar's Feature Selection Algorithm.")

#   num_features = int(input("Please enter total number of features: "))

#   print("Type the number of the algorithm you want to run.")
#   print("\t 1)Forward Selection")
#   print("\t 2)Backward Elimination")

#   algo_choice = int(input())

#   features = list(range(num_features))

#   if algo_choice == 1:
#     print(forward_selection(features))
#   elif algo_choice == 2:
#     print(backward_elimination(features))

def run_classifier_and_validator(dataset, feature_subset):
  classifier = NearestNeighborClassifier()
  validator = Validator()
  accuracy = validator.validate(feature_subset, classifier, dataset)
  return accuracy

# adapted from https://stackoverflow.com/questions/29307532/python-how-can-read-as-float-numbers-a-series-of-strings-from-a-text-file
def load_dataset(filename):
  print("Loading dataset from " + filename + "...")
  start_time = time.time()
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
  print("Time taken: " + str(round((time.time() - start_time) * 1000, 2)) + " ms\n")
  return dataset

def main():
  print("Welcome to Justin Albert, Vinden Drummond, and Sarbesh Sankar's NN classifier and validator.")
  
  print("Type the number of the dataset you want to run.")
  print("\t 1) Small Test Dataset")
  print("\t 2) Large Test Dataset")

  dataset_choice = int(input())
  dataset = None
  # a 1 indexed array of the features to use
  feature_subset = []
  if dataset_choice == 1:
    dataset = load_dataset("small-test-dataset.txt")
    feature_subset = [3, 5, 7]
  elif dataset_choice == 2:
    dataset = load_dataset("large-test-dataset.txt")
    feature_subset = [1, 15, 27]
  
  print("Running NN classifier and validator on dataset...\n")
  # print(dataset)
  accuracy = run_classifier_and_validator(dataset, feature_subset)
  


main()