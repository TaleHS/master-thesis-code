#Training script

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, TimeDistributed
from keras.layers import LSTM, Dropout

import numpy as np
import pandas as pd
import os
import pickle
import math

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt


def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period."""
    filelist = sorted(os.listdir(directory))
    return [x for x in filelist
            if not (x.startswith('.'))]

def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    # ba = a - b
    # bc = c - b
    # if ((c[1]-b[1] <0) | (c[0]-b[0]<0)) & ((a[1]-b[1]<0)|(a[0]-b[0]<0)):
    #     #if (bc.all()<0):
    #     print("both had a negative number")
    #     print("ba: ", ba)
    #     print("bc", bc)
    #print(math.atan2(c[1]-b[1], c[0]-b[0]))
    #print(math.atan2(a[1]-b[1], a[0]-b[0]))
    #return ang + 360 if ang < 0 else ang
    #return ang
    #print("\nThe actual angle: ",ang)
    #return ang + 180 if ang < -180 else ang #should I have elif ang>180: ang-180?
    if ang < -180:
        #ang = ang +180
        #print("ang is less than -180")
        ang = ang +360
    elif ang > 180:
        #ang = ang -180
        ang = ang-360
    return ang


def calc_angle_miss(a, b, c):
    # b always has to be middle point
    ba = a - b
    bc = c - b

    if (b==a).all() | (b==c).all():
        return np.degrees(0.0)

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def get_6_feature_angles_missing_check(X_train):

    x, y, z = X_train.shape
    print(X_train.shape)

    angles = np.zeros((x, y, 6))

    for data in range(x):
        previous_array = np.zeros((z))#np.zeros((1, z))
        for seq in range(y):

            if X_train[data, seq, 2] == 0:
                X_train[data, seq, 2] = previous_array[2]
            if X_train[data, seq, 3] == 0:
                X_train[data, seq, 3] = previous_array[3]
            if X_train[data, seq, 4] == 0:
                X_train[data, seq, 4] = previous_array[4]
            if X_train[data, seq, 5] == 0:
                X_train[data, seq, 5] = previous_array[5]
            if X_train[data, seq, 6] == 0:
                X_train[data, seq, 6] = previous_array[6]
            if X_train[data, seq, 7] == 0:
                X_train[data, seq, 7] = previous_array[7]
            if X_train[data, seq, 8] == 0:
                X_train[data, seq, 8] = previous_array[8]
            if X_train[data, seq, 9] == 0:
                X_train[data, seq, 9] = previous_array[9]
            if X_train[data, seq, 10] == 0:
                X_train[data, seq, 10] = previous_array[10]
            if X_train[data, seq, 11] == 0:
                X_train[data, seq, 11] = previous_array[11]
            if X_train[data, seq, 12] == 0:
                X_train[data, seq, 12] = previous_array[12]
            if X_train[data, seq, 13] == 0:
                X_train[data, seq, 13] = previous_array[13]
            if X_train[data, seq, 14] == 0:
                X_train[data, seq, 14] = previous_array[14]
            if X_train[data, seq, 15] == 0:
                X_train[data, seq, 15] = previous_array[15]
            if X_train[data, seq, 18] == 0:
                X_train[data, seq, 18] == previous_array[18]
            if X_train[data, seq, 19] == 0:
                X_train[data, seq, 19] = previous_array[19]
            if X_train[data, seq, 20] == 0:
                X_train[data, seq, 20] = previous_array[20]
            if X_train[data, seq, 21] == 0:
                X_train[data, seq, 21] = previous_array[21]
            if X_train[data, seq, 22] == 0:
                X_train[data, seq, 22] = previous_array[22]
            if X_train[data, seq, 23] == 0:
                X_train[data, seq, 23] = previous_array[23]
            if X_train[data, seq, 24] == 0:
                X_train[data, seq, 24] = previous_array[24]
            if X_train[data, seq, 25] == 0:
                X_train[data, seq, 25] = previous_array[25]
            if X_train[data, seq, 26] == 0:
                X_train[data, seq, 26] = previous_array[26]
            if X_train[data, seq, 27] == 0:
                X_train[data, seq, 27] = previous_array[27]
            if X_train[data, seq, 28] == 0:
                X_train[data, seq, 28] = previous_array[28]
            if X_train[data, seq, 29] == 0:
                X_train[data, seq, 29] = previous_array[29]

            # ang_right_elbow = calc_angle_miss(np.array((X_train[data, seq, 4], X_train[data, seq, 5])),
            #                              np.array((X_train[data, seq, 6], X_train[data, seq, 7])),
            #                              np.array((X_train[data, seq, 8], X_train[data, seq, 9])))
            # ang_right_shoulder = calc_angle_miss(np.array((X_train[data, seq, 2], X_train[data, seq, 3])),
            #                                 np.array((X_train[data, seq, 4], X_train[data, seq, 5])),
            #                                 np.array((X_train[data, seq, 6], X_train[data, seq, 7])))
            # ang_right_knee = calc_angle_miss(np.array((X_train[data, seq, 18], X_train[data, seq, 19])),
            #                             np.array((X_train[data, seq, 20], X_train[data, seq, 21])),
            #                             np.array((X_train[data, seq, 22], X_train[data, seq, 23])))
            # ang_left_elbow = calc_angle_miss(np.array((X_train[data, seq, 10], X_train[data, seq, 11])),
            #                             np.array((X_train[data, seq, 12], X_train[data, seq, 13])),
            #                             np.array((X_train[data, seq, 14], X_train[data, seq, 15])))
            # ang_left_shoulder = calc_angle_miss(np.array((X_train[data, seq, 2], X_train[data, seq, 3])),
            #                                np.array((X_train[data, seq, 10], X_train[data, seq, 11])),
            #                                np.array((X_train[data, seq, 12], X_train[data, seq, 13])))
            # ang_left_knee = calc_angle_miss(np.array((X_train[data, seq, 24], X_train[data, seq, 25])),
            #                            np.array((X_train[data, seq, 26], X_train[data, seq, 27])),
            #                            np.array((X_train[data, seq, 28], X_train[data, seq, 29])))

            ang_right_elbow = getAngle(np.array((X_train[data, seq, 4], X_train[data, seq, 5])),
                                         np.array((X_train[data, seq, 6], X_train[data, seq, 7])),
                                         np.array((X_train[data, seq, 8], X_train[data, seq, 9])))
            ang_right_shoulder = getAngle(np.array((X_train[data, seq, 2], X_train[data, seq, 3])),
                                            np.array((X_train[data, seq, 4], X_train[data, seq, 5])),
                                            np.array((X_train[data, seq, 6], X_train[data, seq, 7])))
            ang_right_knee = getAngle(np.array((X_train[data, seq, 18], X_train[data, seq, 19])),
                                        np.array((X_train[data, seq, 20], X_train[data, seq, 21])),
                                        np.array((X_train[data, seq, 22], X_train[data, seq, 23])))
            ang_left_elbow = getAngle(np.array((X_train[data, seq, 10], X_train[data, seq, 11])),
                                        np.array((X_train[data, seq, 12], X_train[data, seq, 13])),
                                        np.array((X_train[data, seq, 14], X_train[data, seq, 15])))
            ang_left_shoulder = getAngle(np.array((X_train[data, seq, 2], X_train[data, seq, 3])),
                                           np.array((X_train[data, seq, 10], X_train[data, seq, 11])),
                                           np.array((X_train[data, seq, 12], X_train[data, seq, 13])))
            ang_left_knee = getAngle(np.array((X_train[data, seq, 24], X_train[data, seq, 25])),
                                       np.array((X_train[data, seq, 26], X_train[data, seq, 27])),
                                       np.array((X_train[data, seq, 28], X_train[data, seq, 29])))

            previous_array = X_train[data, seq]

            # Could have had a third for-loop for these 6 lines
            angles[data, seq][0] = ang_right_shoulder
            angles[data, seq][1] = ang_left_shoulder
            angles[data, seq][2] = ang_right_elbow
            angles[data, seq][3] = ang_left_elbow
            angles[data, seq][4] = ang_right_knee
            angles[data, seq][5] = ang_left_knee

    return angles


def load_data_sliding_window_10classes(path=''):
    data_size = len(mylistdir(path))
    xs = []
    Y_all = np.zeros((1, ACTIONS))
    Y_temp = np.zeros((1, ACTIONS))


    for j, ar_name in enumerate(mylistdir(path)):
        #print(ar_name)

        sequence = np.load(path+ar_name)
        seq_size, f, e = sequence.shape
        sequence = np.reshape(sequence, (seq_size, JOINTS*JOINT_FEATURES))


        if 'g01' in ar_name:
            #put true value a01 into y_train
            target = np.zeros(ACTIONS)
            np.put(target,1,1)
            #Y_test[j] = target
            Y_temp[0] = target

        elif 'g02' in ar_name:
            target = np.zeros(ACTIONS)
            np.put(target,2,1)
            Y_temp[0] = target

        elif 'g03' in ar_name:
            target = np.zeros(ACTIONS)
            np.put(target,3,1)
            Y_temp[0] = target

        elif 'g04' in ar_name:
            target = np.zeros(ACTIONS)
            np.put(target,4,1)
            Y_temp[0] = target

        elif 'g06' in ar_name:
            target = np.zeros(ACTIONS)
            np.put(target,5,1)
            #Y_test[j] = target
            Y_temp[0] = target

        elif 'g07' in ar_name:
            target = np.zeros(ACTIONS)
            np.put(target,6,1)
            #Y_test[j] = target
            Y_temp[0] = target

        elif 'g08' in ar_name:
            target = np.zeros(ACTIONS)
            np.put(target,7,1)
            #Y_test[j] = target
            Y_temp[0] = target

        elif 'g09' in ar_name:
            target = np.zeros(ACTIONS)
            np.put(target,8,1)
            #Y_test[j] = target
            Y_temp[0] = target

        elif 'g10' in ar_name:
            target = np.zeros(ACTIONS)
            np.put(target,9,1)
            #Y_test[j] = target
            Y_temp[0] = target

        elif 'g11' in ar_name: # final action
            target = np.zeros(ACTIONS)
            np.put(target,0,1)
            #Y_test[j] = target
            Y_temp[0] = target
        else:
            #print("Unable to determine action label")
            pass

        num_steps = SEQ_LEN
        for i in range(len(sequence) - num_steps + 1):
            example = sequence[i: i + num_steps]
            xs.append(example)
            Y_all = np.append(Y_all, Y_temp, axis=0)

    X_test = np.array(xs)
    Y_all = np.delete(Y_all, 0, 0)
    print("Len of xs: ", len(xs))
    print("Shape of X_test: ", X_test.shape)
    print("Shape of Y: ", Y_all.shape)
    #return X_test, Y_test
    #return xs, Y_all, X_test
    return X_test, Y_all


def load_data_sliding_window_XXclasses(path=''):
    data_size = len(mylistdir(path))
    xs = []
    Y_all = np.zeros((1, ACTIONS))
    Y_temp = np.zeros((1, ACTIONS))


    for j, ar_name in enumerate(mylistdir(path)):
        #print(ar_name)

        sequence = np.load(path+ar_name)
        seq_size, f, e = sequence.shape
        sequence = np.reshape(sequence, (seq_size, JOINTS*JOINT_FEATURES))


        # if 'g01' in ar_name:
        #     #put true value a01 into y_train
        #     target = np.zeros(ACTIONS)
        #     np.put(target,1,1)
        #     #Y_test[j] = target
        #     Y_temp[0] = target
        #
        # if 'g02' in ar_name:
        #     target = np.zeros(ACTIONS)
        #     np.put(target,1,1)
        #     Y_temp[0] = target

        if 'g03' in ar_name:
            target = np.zeros(ACTIONS)
            np.put(target,1,1)
            Y_temp[0] = target

        elif 'g04' in ar_name:
            target = np.zeros(ACTIONS)
            np.put(target,2,1)
            Y_temp[0] = target
        # elif 'g05' in ar_name:
        #     target = np.zeros(ACTIONS)
        #     np.put(target,4,1)
        #     Y_temp[0] = target
        # elif 'g06' in ar_name:
        #     target = np.zeros(ACTIONS)
        #     np.put(target,4,1)
        #     #Y_test[j] = target
        #     Y_temp[0] = target

        elif 'g07' in ar_name:
            target = np.zeros(ACTIONS)
            np.put(target,3,1)
            #Y_test[j] = target
            Y_temp[0] = target

        elif 'g08' in ar_name:
            target = np.zeros(ACTIONS)
            np.put(target,4,1)
            #Y_test[j] = target
            Y_temp[0] = target

        elif 'g09' in ar_name:
            target = np.zeros(ACTIONS)
            np.put(target,5,1)
            #Y_test[j] = target
            Y_temp[0] = target

        elif 'g10' in ar_name:
            target = np.zeros(ACTIONS)
            np.put(target,6,1)
            #Y_test[j] = target
            Y_temp[0] = target

        elif 'g11' in ar_name: # final action
            target = np.zeros(ACTIONS)
            np.put(target,0,1)
            #Y_test[j] = target
            Y_temp[0] = target
        elif 'g13' in ar_name: # final action
            target = np.zeros(ACTIONS)
            np.put(target,7,1)
            #Y_test[j] = target
            Y_temp[0] = target
        else:
            print("\nUnable to determine action label")
            pass

        num_steps = SEQ_LEN
        for i in range(len(sequence) - num_steps + 1):
            example = sequence[i: i + num_steps]
            xs.append(example)
            Y_all = np.append(Y_all, Y_temp, axis=0)

    X_test = np.array(xs)
    Y_all = np.delete(Y_all, 0, 0)
    print("Len of xs: ", len(xs))
    print("Shape of X_test: ", X_test.shape)
    print("Shape of Y: ", Y_all.shape)
    #return X_test, Y_test
    #return xs, Y_all, X_test
    return X_test, Y_all



if __name__ == "__main__":
    # Data parameters (data shape for each sample = (SEQ_LEN, JOINTS, JOINT_FEATURES)):
    ACTIONS = 8#9#10#7#10#11
    SEQ_LEN = 8#5#8#10
    JOINTS = 25
    JOINT_FEATURES = 2
    features = JOINTS*JOINT_FEATURES


    HIDDEN_UNITS = 256 # try out different number of units.
    BATCH_SIZE = 32#16 # BS=8 gives 11 samples at each batch
    EPOCHS = 200
    DROPOUT = 0.2
    LR = 1e-4 # learning rate
    OPTIMIZER = keras.optimizers.RMSprop(lr=LR)


    X_train, Y_train = load_data_sliding_window_XXclasses("/path_to_training_data/np_train/")
    X_val, Y_val = load_data_sliding_window_XXclasses("/path_to_validation_data/np_val/")
    X_test, Y_test = load_data_sliding_window_XXclasses("/path_to_test_data/np_test/")



    ang_train = get_6_feature_angles_missing_check(X_train)
    ang_val = get_6_feature_angles_missing_check(X_val)
    ang_test = get_6_feature_angles_missing_check(X_test)


    # # Normalising the angles between -1 and 1
    # # Squish into 2d bs sklearn is stupid
    # feat = 6
    # angles_train_norm = np.reshape(ang_train, (ang_train.shape[0], SEQ_LEN*feat))
    # angles_val_norm = np.reshape(ang_val, (ang_val.shape[0], SEQ_LEN*feat))
    # angles_test_norm = np.reshape(ang_test, (ang_test.shape[0], SEQ_LEN*feat))
    #
    # print("Shape of training data during standarisation: ", angles_train_norm.shape)
    #
    # #Standardisiation (Z-score)
    # scaler = preprocessing.StandardScaler()
    # angles_train_norm = scaler.fit_transform(angles_train_norm)
    # angles_val_norm = scaler.transform(angles_val_norm)
    # angles_test_norm = scaler.fit_transform(angles_test_norm)
    #
    # # Reshape back to 3D
    # ang_train = np.reshape(angles_train_norm, (angles_train_norm.shape[0], SEQ_LEN, feat))
    # ang_val = np.reshape(angles_val_norm, (angles_val_norm.shape[0], SEQ_LEN, feat))
    # ang_test = np.reshape(angles_test_norm, (angles_test_norm.shape[0], SEQ_LEN, feat))
    #
    # # print("Shape of training data after standarisation: ", angles_train_norm.shape)
    # print("Shape of training data after standarisation: ", ang_train.shape)


    feat = 6
    model_train = Sequential()
    model_train.add(LSTM(HIDDEN_UNITS, input_shape=(SEQ_LEN, feat),return_sequences=True, dropout=DROPOUT))
    model_train.add(LSTM(HIDDEN_UNITS, dropout=DROPOUT))
    model_train.add(Dense(ACTIONS, activation='softmax'))
    model_train.compile(loss='categorical_crossentropy', optimizer=OPTIMIZER, metrics=['acc'])

    #callback lagrer weight underveis slik at vi kan loade weights inn i ny model som tar sequence length of 1 istedet for 83
    #history = model_train.fit(angles_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(angles_val, Y_val))
    call_path = '/directory_path/model.{epoch:02d}-{val_acc:.2f}.h5'


    #Saving the model to a directory
    my_callbacks = keras.callbacks.ModelCheckpoint(filepath= call_path)

    history = model_train.fit(ang_train, Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[my_callbacks], validation_data=(ang_val, Y_val))

    #Evaluating the model
    score = model_train.evaluate(ang_test, Y_test)
    print("Test set loss: ", score[0])
    print("Test set accuracy: ", score[1])
