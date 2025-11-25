from search_algos import forward_selection

def main():
  
  print("Welcome to Justin Albert, Vinden Drummond, and Sarbesh Sankar's Feature Selection Algorithm.")

  num_features = int(input("Please enter total number of features: "))

  print("Type the number of the algorithm you want to run.")
  print("\t 1)Forward Selection")
  print("\t 2)Backward Elimination")

  algo_choice = int(input())

  features = list(range(num_features))

  if algo_choice == 1:
    print(forward_selection(features))
  elif algo_choice == 2:
    print(backward_elimination(features))
  else:
    print("Invalid choice")
    return 0


main()