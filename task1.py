import numpy as np
import math

class Node(object):
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_child(self, obj):
        self.children.append(obj)
    
    def is_leaf(self):
        if len(self.children) == 0:
            return True
        else:
            return False


def calc_conditional_entropy(data, classes):
    conditional_entropy = 0
    nData = len(data)
    # Find which values can we have in the dataset
    values = []
    for datapoint in data:
        if datapoint not in values:
            values.append(datapoint)
    featureCounts = np.zeros(len(values))
    entropy = np.zeros(len(values))
    valueIndex=0

    # Count how many times each value occurs in the dataset
    for value in values:
        dataIndex = 0
        newClasses = []
        for datapoint in data:
            if datapoint == value:
                featureCounts[valueIndex]+=1
                newClasses.append(classes[dataIndex])
            dataIndex += 1
        # Get the values in newClasses
        classValues = []
        for aclass in newClasses:
            if classValues.count(aclass)==0:
                classValues.append(aclass)
        classCounts = np.zeros(len(classValues))
        classIndex = 0
        for classValue in classValues:
            for aclass in newClasses:
                if aclass == classValue:
                    classCounts[classIndex]+=1
            classIndex += 1
        for classIndex in range(len(classValues)):
           p = float(classCounts[classIndex])/sum(classCounts)
           entropy[valueIndex] += -(p * np.log2(p))
        conditional_entropy += float(featureCounts[valueIndex])/nData * entropy[valueIndex]
        valueIndex+=1
    return conditional_entropy

def calc_entropy(data):
    calced_entropy = 0
    # Find which values can we have in the dataset
    values = []
    for datapoint in data:
        if datapoint not in values:
            values.append(datapoint)
    featureCounts = np.zeros(len(values))
    entropy = np.zeros(len(values))
    valueIndex=0

    # Count how many times each value occurs in the dataset
    for value in values:
        for datapoint in data:
            if datapoint == value:
                featureCounts[valueIndex]+=1

        
        p = float(featureCounts[valueIndex])/len(data)
        entropy[valueIndex] += -(p * np.log2(p))
        calced_entropy += entropy[valueIndex]
        valueIndex+=1    
    return calced_entropy

    # Calculate the entropy of the dataset

def calc_info_gain(data, classes):
    return calc_entropy(classes)-calc_conditional_entropy(data, classes)

def highest_info_gain(X, y):
    data_keys = X.keys()
    largest_gain = 0
    largest_title = ""
    for key in data_keys:
        info_gain = calc_info_gain(X[key], y)
        if info_gain > largest_gain:
            largest_gain = info_gain
            largest_title = key
    return largest_title

def all_data_points_same_label(X, y):
    return False

def all_data_points_identical_feature():
    return False

def build_DT_recursively(X,y):
    return False

def learn(X, y, impurity_measure='entropy'):
    if X:
        key_of_highest_info_gain = highest_info_gain(X, y['Activity'])
        root = Node(key_of_highest_info_gain)
        copy_X = X
        del copy_X[key_of_highest_info_gain]
        print(key_of_highest_info_gain)
        print(copy_X)
        root.add_child(learn(copy_X, y))
        return root
    else:
        return

#    entropy = 0.0
#    for featureCount in featureCounts:
#        entropy -= featureCount/len(data) * float(np.log2(featureCount/len(data)))
#    return entropy
    

#def calc_info_gain(data, target):
#    return calc_entropy(target) - calc_entropy(data)



#def predict(x, tree):



# training data
train_deadline = ['Urgent', 'Urgent', 'Near', 'None', 'None', 'None', 'Near', 'Near', 'Near', 'Urgent']
train_party = ['Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'No']
train_lazy = ['Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'No', 'Yes', 'Yes', 'No']
train_activity = ['Party', 'Study', 'Party', 'Party', 'Pub', 'Party', 'Study', 'TV', 'Party', 'Study']


X = {'Deadline': train_deadline, 'Party': train_party, 'Lazy': train_lazy}
y = {'Activity': train_activity}

#calc_conditional_entropy(X['Deadline'], y['Activity'])
#calc_entropy(y['Activity'])
#print(calc_info_gain(X['Deadline'], y['Activity']))
#print(highest_info_gain(X, y['Activity']))
learn(X, y)