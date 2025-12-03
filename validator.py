class Validator:
    def validate(self, feature_subset, classifier, dataset):
        # leave-one-out cross-validation
        correct_predictions = 0
        total_instances = len(dataset)
        
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
                for feat_idx in feature_subset:  # feature_subset is 1-indexed
                    feature_vector.append(instance[feat_idx])
                training_tuples.append((feature_vector, class_label))
            
            classifier.train(training_tuples)
            
            # extract features from test instance
            test_features = []
            for feat_idx in feature_subset:
                test_features.append(test_instance[feat_idx])
            
            predicted_class = classifier.test(test_features)
            
            if predicted_class == test_class:
                correct_predictions += 1
        
        return correct_predictions / total_instances

