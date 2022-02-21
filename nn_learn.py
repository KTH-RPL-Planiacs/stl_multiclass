import numpy as np
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from evaluation_metrics import hamming_loss
from evaluation_metrics import example_based_accuracy
import time
from sklearn.metrics import confusion_matrix
import time


def plot_cm(y_true, y_pred, class_names):
    y_true = [str(list(i)) for i in y_true]
    y_pred = [str(list(i)) for i in y_pred]
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 9)) 
    ax = sns.heatmap(
            cm, 
            annot=True, 
            fmt="d", 
            cmap=sns.diverging_palette(220, 20, n=7),
            ax=ax
    )

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    b, t = plt.ylim() # discover the values for bottom and top
    b += 0.5 # Add 0.5 to the bottom
    t -= 0.5 # Subtract 0.5 from the top
    plt.ylim(b, t) # update the ylim(bottom, top) values
    plt.show() # ta-da!







def learn_nn(X_train, y_train):

    #from https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py
    # model = keras.Sequential()
    # model.add(
        # keras.layers.Bidirectional(
          # keras.layers.LSTM(
              # units=256, 
              # input_shape=[X_train.shape[1], X_train.shape[2]]
          # )
        # )
    # )
    # model.add(keras.layers.Dropout(rate=0.5))
    # model.add(keras.layers.Dense(units=256, activation='relu'))
    # model.add(keras.layers.Dense(y_train.shape[1], activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])


    #from https://machinelearningmastery.com/multi-label-classification-with-deep-learning/
    # model = Sequential()
	# model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	# model.add(Dense(n_outputs, activation='sigmoid'))
	# model.compile(loss='binary_crossentropy', optimizer='adam')

    
    #our model
    model = keras.Sequential()   
    model.add(
        keras.layers.Bidirectional(
          keras.layers.LSTM(
              units=256, 
              input_shape=[X_train.shape[1], X_train.shape[2]]
          )
        )
    )    
    model.add(keras.layers.Dense(units=512, activation='sigmoid'))
    model.add(keras.layers.Dense(units=256, activation='sigmoid'))
    model.add(keras.layers.Dense((y_train.shape[1]), activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam')
    
    
    model.fit(
        X_train, y_train,
        epochs=20,
        batch_size=128,
        validation_split=0.1,
        shuffle=True
    )
    
    return model
    

    
    
    

    
if __name__ == '__main__':

    dict_trajectories_classes = pickle.load(open("processed_classes/dict_trajectories_classes.pkl", "rb" ))
    dict_trajectories = pickle.load(open("processed_classes/dict_trajectories.pkl", "rb" ))
    dict_classes_trajectory = pickle.load(open("processed_classes/dict_classes_trajectory.pkl", "rb" ))
    list_classes = pickle.load(open("processed_classes/list_classes.pkl", "rb" ))


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
        
    X_id    = np.array(X_id)
    y_train = np.array(y)

    X_train = np.array([np.array(dict_trajectories[t])[:125] for t in X_id])
    
    start_time = time.time()
    model = learn_nn(X_train, y_train)
    end_time   = time.time()
    print("time(s)",end_time-start_time)
    
    model.save('output_models/nn_model.pkl')
    model = keras.models.load_model('output_models/nn_model.pkl')
    
    #Evaluate on the train set
    y_pred = model.predict(X_train)
    y_pred = np.array([[int(prob>0.4) for prob in y] for y in y_pred])
    # y_pred = y_pred.round().astype(int)
    print("trainset: ",hamming_loss(np.array(y_train), np.array(y_pred)),example_based_accuracy(np.array(y_train), np.array(y_pred)))
    
    class_names = np.unique([str(list(i)) for i in list(np.concatenate((y_train, y_pred), axis=0))])
    plot_cm(y_train, y_pred, class_names)
    