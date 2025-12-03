import time


class Validator:
    # dataset is a list of arrays where (class_lable, feature_vector)
    # feature_subset is a list of feature indexes
    def validate(self, feature_subset, classifier, dataset):
        print("Validating with feature subset: {"+ ", ".join(map(str, feature_subset))+ "}\n")
        # leave-one-out cross-validation
        correct_predictions = 0
        total_instances = len(dataset)

        start_time = time.time()  # Record start time in milliseconds
        start_time_50 = time.time() * 1000

        for i in range(total_instances):
            test_instance = dataset[i]
            test_class = test_instance[0]

            # use all instances except i for training
            training_data = []
            for j in range(total_instances):
                if j != i:
                    training_data.append(dataset[j])

            # extract features and convert to (feature_vector, class_label) format
            training_tuples = []
            for instance in training_data:
                class_label = instance[0]
                feature_vector = []
                for feat_idx in feature_subset:
                    feature_vector.append(instance[1][feat_idx - 1])
                training_tuples.append((feature_vector, class_label))

            classifier.train(training_tuples)

            # extract features from test instance
            test_features = []
            for feat_idx in feature_subset:
                test_features.append(test_instance[1][feat_idx - 1])

            predicted_class = classifier.test(test_features)

            if predicted_class == test_class:
                correct_predictions += 1

            # print the time every 50 rows
            if (i + 1) % 50 == 0:
                end_time_50 = time.time() * 1000
                batch_time = end_time_50 - start_time_50
                print("Time for rows " + str((i - 48)) + " to " + str(i + 1) + " - " + str(round(batch_time, 2)) + " ms")
                start_time_50 = time.time() * 1000

        print("\n")
        print("Correct predictions: " + str(correct_predictions))
        print("Total instances: " + str(total_instances))
        print("Accuracy: " + str(correct_predictions / total_instances * 100) + "%")
        print("Total time taken: " + str(round((time.time() - start_time) * 1000, 2)) + " ms")

        return correct_predictions / total_instances
