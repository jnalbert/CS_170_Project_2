import math

class NearestNeighborClassifier:
    
    def __init__(self):
        # list of tuples (feature_vector, class_label)
        self.training_instances = []
    
    def train(self, training_data):
        self.training_instances = training_data
    
    # pass in a feature vector and return the class label
    def test(self, test_instance):
        min_distance = float('inf')
        nearest_class = None
        
        for train_features, train_class in self.training_instances:
            # n dimension euclidean distance
            squared_sum = 0.0
            for i in range(len(test_instance)):
                diff = test_instance[i] - train_features[i]
                squared_sum += (diff * diff)
            distance = math.sqrt(squared_sum)
            
            if distance < min_distance:
                # if distance is less than our current min, replace our saved min value
                min_distance = distance
                nearest_class = train_class
                
        return nearest_class
