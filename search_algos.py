from nn_classifier import NearestNeighborClassifier
from validator import Validator

classifier = NearestNeighborClassifier()
validator = Validator()


# features is a list of integers
def evaluation_function(current_features, dataset):
    accuracy = validator.validate(current_features, classifier, dataset)
    return accuracy


def format_feature_set(features):
    feature_strings = []
    for f in features:
        feature_strings.append(str(f))
    return "{" + ",".join(feature_strings) + "}"


# features is a list of integers
# dataset is a list of tuples where each tuple is (class_label, feature_vector)
def forward_selection(features, dataset):
    no_features_accuracy = evaluation_function([], dataset)
    print(
        'Running nearest neighbor with no features (default rate), using "leaving-one-out" evaluation, I get an accuracy of '
        + str(round(no_features_accuracy * 100, 1))
        + "%"
    )
    print("\nBeginning search.\n")

    curr_best_features = []
    curr_best_accuracy = no_features_accuracy
    # we make a copy since we don't actually want to modify the original list
    features_left = features.copy()

    # go through all of the features one by one, and try and add them to our current best set
    # this is our operator or sorts
    while features_left:
        best_feature = None
        best_accuracy = -1
        best_set = None

        for feature in features_left:
            new_feature_set = [feature] + curr_best_features
            accuracy = evaluation_function(new_feature_set, dataset)

            feature_str = format_feature_set(new_feature_set)
            print(
                "Using feature(s) "
                + feature_str
                + " accuracy is "
                + str(round(accuracy * 100, 1))
                + "%"
            )

            # if adding the current feature leads to an accuracy better than our current best then we should use that one
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature = feature
                best_set = new_feature_set

        # now that weve found the best feature to add to this set we'll check if the accruacy is better than our current best
        if best_accuracy < curr_best_accuracy:
            # if the accruate decreased then we should stop and return our current best set of features
            print("\n(Warning, Accuracy has decreased!)\n")
            break

        curr_best_features = best_set
        curr_best_accuracy = best_accuracy
        features_left.remove(best_feature)

        feature_str = format_feature_set(curr_best_features)
        print(
            "Feature set "
            + feature_str
            + " was best, accuracy is "
            + str(round(curr_best_accuracy * 100, 1))
            + "%"
        )
        print()

    final_feature_str = format_feature_set(curr_best_features)
    print(
        "Finished search!! The best feature subset is "
        + final_feature_str
        + ", which has an accuracy of "
        + str(round(curr_best_accuracy * 100, 1))
        + "%"
    )

    return curr_best_features


# features is a list of integers
# dataset is a list of tuples where each tuple is (class_label, feature_vector)
def backward_elimination(features, dataset):
    # start with all features
    all_features_accuracy = evaluation_function(features, dataset)
    print(
        'Running nearest neighbor with all features, using "leaving-one-out" evaluation, I get an accuracy of '
        + str(round(all_features_accuracy * 100, 1))
        + "%"
    )
    print("\nBeginning search.\n")
    # make a copy so original list isnt modified
    curr_best_features = features.copy()
    curr_best_accuracy = all_features_accuracy

    # while we have more than one feature, try removing one at a time
    while len(curr_best_features) > 1:
        best_feature_to_remove = None
        best_accuracy = -1
        best_set = None

        # try removing each feature that is in the list 
        for feature in curr_best_features:
            new_feature_set = curr_best_features.copy()
            new_feature_set.remove(feature)

            accuracy = evaluation_function(new_feature_set, dataset)

            feature_str = format_feature_set(new_feature_set)
            print(
                "Using feature(s) "
                + feature_str
                + " accuracy is "
                + str(round(accuracy * 100, 1))
                + "%"
            )

            # after removing one feature, keep track of best accuracy subset
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature_to_remove = feature
                best_set = new_feature_set

        # if removing the feature decreases the acurracy then stop
        if best_accuracy < curr_best_accuracy:
            print("\n(Warning, Accuracy has decreased!)\n")
            break

        # otherwise, update the current best subset
        curr_best_features = best_set
        curr_best_accuracy = best_accuracy

        feature_str = format_feature_set(curr_best_features)
        print(
            "Removing "
            + str(best_feature_to_remove)
            + ". Feature set "
            + feature_str
            + " was best, accuracy is "
            + str(round(curr_best_accuracy * 100, 1))
            + "%"
        )
        print()

    final_feature_str = format_feature_set(curr_best_features)
    print(
        "Finished search!! The best feature subset is "
        + final_feature_str
        + ", which has an accuracy of "
        + str(round(curr_best_accuracy * 100, 1))
        + "%"
    )

    return curr_best_features