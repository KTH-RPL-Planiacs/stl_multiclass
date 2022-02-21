import numpy as np
import math

def hamming_loss(y_true, y_pred):
    """
    XOR TT for reference - 
    
    A  B   Output
    
    0  0    0
    0  1    1
    1  0    1 
    1  1    0
    """
    hl_num = np.sum(np.logical_xor(y_true, y_pred))
    hl_den = np.prod(y_true.shape)
    
    return hl_num/hl_den

def example_based_accuracy(y_true, y_pred):
    # compute true positives using the logical AND operator
    numerator = np.sum(np.logical_and(y_true, y_pred), axis = 1)

    # compute true_positive + false negatives + false positive using the logical OR operator
    denominator = np.sum(np.logical_or(y_true, y_pred), axis = 1)
    # print('\t',denominator)
    instance_accuracy = numerator/denominator

    avg_accuracy = np.mean(instance_accuracy)
    return avg_accuracy
    
    
def log(x):
    if x == 0:
        return 0
    else:
        return math.log(x,2)


def multi_label_entropy(dict_trajectories_classes,list_classes,set_trajectories):
    num_classes = {class_i:0 for class_i in list_classes}
    for trajectory in set_trajectories:
        for class_i in list(dict_trajectories_classes[trajectory]):
            num_classes[class_i] += 1
    freq_classes = {class_i:num_classes[class_i]/len(set_trajectories) for class_i in list_classes}
    ent = 0
    for num in freq_classes:
        ent += (freq_classes[num]*log(freq_classes[num])) + ( (1-freq_classes[num]) * log(1-freq_classes[num]) )
    return ent*-1
    

if __name__ == "__main__":
    y_true = np.array([[0,1,0],[0,0,1],[0,1,0],[0,1,0],[1,0,1],[1,1,0],[1,0,0],[1,0,0],[0,1,0],[1,0,1]])
    y_pred = np.array([[0,1,1],[1,1,1],[0,1,0],[0,1,0],[1,0,1],[1,1,0],[1,0,0],[1,0,0],[0,1,0],[1,0,1]])
    # y_pred = np.array([[1,1,1],[1,1,0],[1,0,1],[1,0,1],[0,1,0],[0,0,1],[0,1,1],[1,1,1],[1,0,1],[0,1,0]])
    hl_value = hamming_loss(y_true, y_pred)
    print(f"Hamming Loss: {hl_value}")
    
    ex_based_accuracy = example_based_accuracy(y_true, y_pred)
    print(f"Example Based Accuracy: {ex_based_accuracy}")
    
    
    set_trajectories = [0,1,2,3,4,5,6,7,8,9,10]
    dict_trajectories_classes = {0:(0,),1:(0,1),2:(0,1,2),3:(1,),4:(1,2),5:(2,),6:(0,1),7:(0,1,2),8:(0,1),9:(0,),10:(1,)}
    list_classes = [0,1,2]
    print(multi_label_entropy(dict_trajectories_classes,list_classes,set_trajectories))