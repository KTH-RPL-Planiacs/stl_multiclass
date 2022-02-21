import numpy as np
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import dill
from evaluation_metrics import hamming_loss, example_based_accuracy
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from nn_learn import learn_nn
from multilabel_stl_learn import DTLearn, evaluate
import time



#Loading dataset of multilabel-multiclass trajectories
dict_trajectories_classes = pickle.load(open("synthetic_data/processed_classes/dict_trajectories_classes.pkl", "rb" ))
dict_trajectories = pickle.load(open("synthetic_data/processed_classes/dict_trajectories.pkl", "rb" ))
dict_classes_trajectory = pickle.load(open("synthetic_data/processed_classes/dict_classes_trajectory.pkl", "rb" ))
list_classes = pickle.load(open("synthetic_data/processed_classes/list_classes.pkl", "rb" ))

X_id = []
y    = []
for trajectory in dict_trajectories_classes:
    true = []
    for i in range(0,len(list_classes)):
        if list_classes[i] in dict_trajectories_classes[trajectory]:
            true.append(1)
        else:
            true.append(0)
    y.append(true)
    X_id.append(trajectory)
    
X_id = np.array(X_id)
y    = np.array(y)



#Do a Stratified Shuffle Split cross validation 5 folds (80% train, 20% test)
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
fold = 1

hamming_losses_nn           = []
example_based_accuracies_nn = []
times_nn                    = []
hamming_losses_dt           = []
example_based_accuracies_dt = []
times_dt                    = []

for train_index, test_index in sss.split(X_id, y):

    print("\n\nFOLD:", fold, "TRAIN:", train_index, "TEST:", test_index)
    
    X_id_train, X_id_test = X_id[train_index], X_id[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    #for the NN (125 first time steps of trajectories)
    X_train = np.array([np.array(dict_trajectories[t]) for t in X_id_train])
    X_test  = np.array([np.array(dict_trajectories[t]) for t in X_id_test])
    
    #learn NN model
    start_time = time.time()
    nn_model   = learn_nn(X_train, y_train)
    end_time   = time.time()
    
    nn_model.save('synthetic_data/cv/nn/nn_model_'+str(fold)+'.pkl')
    with open('synthetic_data/cv/nn/nn_model_time_'+str(fold)+'.pkl', "wb") as pfile:
        pickle.dump(end_time-start_time, pfile)
    
    
    nn_model = keras.models.load_model('synthetic_data/cv/nn/nn_model_'+str(fold)+'.pkl')
    total_time = pickle.load(open('synthetic_data/cv/nn/nn_model_time_'+str(fold)+'.pkl', "rb" ))
    

    #Evaluate NN model on the test set
    y_pred = nn_model.predict(X_test)
    y_pred = y_pred.round().astype(int)
    
    hamming_loss_nn           = hamming_loss(np.array(y_test), np.array(y_pred))
    example_based_accuracy_nn = example_based_accuracy(np.array(y_test), np.array(y_pred))
    print("hamming_loss_nn: ", round(hamming_loss_nn,5) ,"example_based_accuracy_nn: ",round(example_based_accuracy_nn,5),"time (s): ",round(end_time-start_time,2))

    hamming_losses_nn.append(hamming_loss_nn)
    example_based_accuracies_nn.append(example_based_accuracy_nn)
    times_nn.append(end_time-start_time)


    #Now for the STL multilabel-multiclass Decision tree
    start_time = time.time()
    max_depth, min_x, min_y, min_h, max_x, max_y, max_h = 5, -8, -8, 10, 8, 8, 100
    dtlearn = DTLearn(dict_trajectories, dict_trajectories_classes, list_classes, min_x, min_y, min_h, max_x, max_y, max_h, max_depth, True)
    tree = dtlearn.recursiveGenerateTree(X_id_train)
    end_time   = time.time()
    
    with open('synthetic_data/cv/stl_dt/stl_dt_'+str(fold)+'.pkl', "wb") as pfile:
        pickle.dump(end_time-start_time, pfile)
    with open("synthetic_data/cv/stl_dt/stl_dt_"+str(fold)+".dill", "wb") as pfile:
        dill.dump(tree, pfile)
    
    
    # tree = dill.load(open("synthetic_data/cv/stl_dt/stl_dt_"+str(fold)+".dill", "rb" ))
    # total_time = pickle.load(open('synthetic_data/cv/nn/nn_model_time_'+str(fold)+'.pkl', "rb" ))

    y_true, y_pred = evaluate(tree, {key: dict_trajectories_classes[key] for key in X_id_test} , {key: dict_trajectories[key] for key in X_id_test}, list_classes)
    
    hamming_loss_dt           = hamming_loss(np.array(y_true), np.array(y_pred))
    example_based_accuracy_dt = example_based_accuracy(np.array(y_true), np.array(y_pred))
    print("hamming_loss_dt: ", round(hamming_loss_dt,5) ,"example_based_accuracy_dt: ",round(example_based_accuracy_dt,5),"time (s): ",round(end_time-start_time,2))

    hamming_losses_dt.append(hamming_loss_dt)
    example_based_accuracies_dt.append(example_based_accuracy_dt)
    times_dt.append(end_time-start_time)
    
    fold += 1



    
print(hamming_losses_nn)
print(example_based_accuracies_nn)
print(times_nn)
print(hamming_losses_dt)
print(example_based_accuracies_dt)
print(times_dt)