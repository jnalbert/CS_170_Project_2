from search_algos import forward_selection

def main():
  num_features = 4
  
  # create a list of fake features which is just 0 - num_features
  features = list(range(num_features))
  print(forward_selection(features))
  return 0

main()