#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re
import numpy as np
import glob
import os
import csv
import itertools
import matplotlib.pyplot as plt


from keras.metrics import top_k_categorical_accuracy
def top_3_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=3)

## From : https://github.com/DantesLegacy/TensorFlow_AudioSet_Example/blob/4ff964048ee528acb685e2af5d941353446a044b/src/neural_network_audioset.py
## Importante! Esta função está assumindo que existem um header de 3 linhas em cada CSV
def get_file_name_labels_from_audioset_csv(row_num,csv_file,audioset_indices_csv):
    str_labels = []
    int_labels = []
    # Open choosen CSV file
    with open(csv_file, 'r') as f:
        # Skip to the line we need.
        line = next(itertools.islice(csv.reader(f), int(row_num) + 3, None))
        #print("line:",line)
        # Now that we have the line we need, we need to grab the labels from it
        # This file may have multiple labels, so we need to account for that
        for element in line[3:]:
            if (element.startswith(' "')) and (element.endswith('"')):
                str_labels.append(element[2:-1])
            elif element.startswith(' "'):
                str_labels.append(element[2:])
            elif element.endswith('"'):
                str_labels.append(element[:-1])
            else:
                str_labels.append(element)

    # Now we have the string version of the labels.
    # Let's convert them to int versions
    for element in str_labels:
        with open(audioset_indices_csv, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if row[1] == element:
                    int_labels.append(int(row[0]))

    return int_labels



def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)



def k_hot_encode(labels,n_unique_labels):
    n_labels = len(labels)
    k_hot_encode = np.zeros((n_labels,n_unique_labels))
    # Mark the relevant values in the area as '1'
    #  This can be multiple elements in the array as there can be
    #  multiple labels to a sample
    for index in range(n_labels):
        for element in labels[index]:
            #print(index,element)
            k_hot_encode[index, element] = 1
    return k_hot_encode


def assure_path_exists(path):
    mydir = os.path.join(os.getcwd(), path)
    if not os.path.exists(mydir):
        os.makedirs(mydir)

def save_files(data_dir,features,labels,save_h5 = False):
    labels = k_hot_encode(labels,n_unique_labels = 7)

    print "Features of = ", features.shape
    print "Labels of = ", labels.shape

    if save_h5:
        feature_file = os.path.join(data_dir + '_x.hdf5')
        labels_file = os.path.join(data_dir + '_y.hdf5')
        with h5py.File(feature_file, 'w') as hf:
            hf.create_dataset("features",  data=features,compression="gzip", compression_opts=9)
        with h5py.File(labels_file, 'w') as hf:
            hf.create_dataset("labels",  data=labels,compression="gzip", compression_opts=9)
    else:
        feature_file = os.path.join(data_dir + '_x.npy')
        labels_file = os.path.join(data_dir + '_y.npy')
        np.save(feature_file, features)
        np.save(labels_file, labels)

    print "Saved " + feature_file
    print "Saved " + labels_file



from sklearn.metrics import  f1_score, precision_score, recall_score,hamming_loss


from keras.callbacks import Callback
class custom_metrics(Callback):
    def on_train_begin(self, logs={}):
        self.custom_metrics = {}
        self.custom_metrics['val_f1s'] = []
        self.custom_metrics['val_recalls'] = []
        self.custom_metrics['val_precisions'] = []
        self.custom_metrics['val_hamming_loss'] = []


    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        _val_f1 = f1_score(val_targ, val_predict,average='micro')
        _val_recall = recall_score(val_targ, val_predict,average='micro')
        _val_precision = precision_score(val_targ, val_predict,average='micro')
        _val_hamming_loss = hamming_loss(val_targ, val_predict)
        self.custom_metrics['val_f1s'].append(_val_f1)
        self.custom_metrics['val_recalls'].append(_val_recall)
        self.custom_metrics['val_precisions'].append(_val_precision)
        self.custom_metrics['val_hamming_loss'].append(_val_hamming_loss)
        #print " — val_f1: %f — val_precision: %f — val_recall %f — val_hamming_loss %f" %(_val_f1, _val_precision, _val_recall,_val_hamming_loss)

        return





def multilabel_confusion_matrix(eval_y,predictions,n_classes):
    """
     Compute True positive,  True negative, False positive,False negative
     for a multilabel classification problem

    https://github.com/scikit-learn/scikit-learn/issues/3452
    http://www.cnts.ua.ac.be/~vincent/pdf/microaverage.pdf
    """

    def check_predicted_labels(label_no,predictions):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for idx, val in enumerate(predictions):
            if(val[label_no] == 1 and eval_y[idx][label_no] == 1):
                TP += 1
            elif(val[label_no] == 0 and eval_y[idx][label_no] == 0):
                TN += 1
            elif(val[label_no] == 1 and eval_y[idx][label_no] == 0):
                FP += 1
            elif(val[label_no] == 0 and eval_y[idx][label_no] == 1):
                FN += 1
        return(TP, FP, TN, FN)


    print("Multilabel Confusion Matrix")
    print("  TP,   FP,     TN,     FN, ")
    predicted_matrix = np.empty((0,4),dtype=int)
    for i in range(n_classes):
        TP,FP, TN, FN = check_predicted_labels(i,predictions)
        temp = np.hstack([TP,FP,TN,FN])
        predicted_matrix = np.vstack([predicted_matrix,temp])

    for idx in range(n_classes):
        print idx,('\t'.join(map(str,predicted_matrix[idx])))
    print "Σ",('\t'.join(map(str,predicted_matrix.sum(axis=0))))
    print("")
    print("F1 Score: %f"%f1_score(eval_y, predictions,average='micro'))
    print("Recall: %f"%recall_score(eval_y, predictions,average='micro'))
    print("Precision: %f"%precision_score(eval_y, predictions,average='micro'))
    print("Hamming Loss: %f"%hamming_loss(eval_y, predictions))


def plot_history(hist):
    print "History keys:", (hist.history.keys())
     #summarise history for training and validation set accuracy

    if ('val_loss' in hist.history):
        for key in hist.history.keys():
            if key[:4] == "val_":
                continue
            elif(key == "lr"):
                continue
            else:
                plt.subplot()
                plt.plot(hist.history[key])
                plt.plot(hist.history['val_%s'%key])
                plt.title('Model %s'%key)
                plt.ylabel(key)
                plt.xlabel('epoch')
                plt.legend(['train', 'validation'], loc='upper left')
                plt.show()
    else:
        for key in hist.history.keys():
            plt.subplot()
            plt.plot(hist.history[key])
            plt.title('Model %s'%key)
            plt.ylabel(key)
            plt.xlabel('epoch')
            plt.legend(['train'], loc='upper left')
            plt.show()


def plot_metrics(metrics):
    for key in metrics.custom_metrics.keys():
        plt.subplot()
        plt.plot(metrics.custom_metrics[key])
        plt.title('Model %s'%key)
        plt.ylabel(key)
        plt.xlabel('epoch')
        plt.legend(['validation'], loc='upper left')
        plt.show()









# From:  https://github.com/philipperemy/keras-visualize-activations
def get_activations(model, model_inputs, print_shape_only=True, layer_name=None):
    import keras.backend as K
    print('----- activations -----')
    activations = []
    inp = model.input

    model_multi_inputs_cond = True
    if not isinstance(inp, list):
        # only one input! let's wrap it in a list.
        inp = [inp]
        model_multi_inputs_cond = False

    outputs = [layer.output for layer in model.layers if
               layer.name == layer_name or layer_name is None]  # all layer outputs

    funcs = [K.function(inp + [K.learning_phase()], [out]) for out in outputs]  # evaluation functions

    if model_multi_inputs_cond:
        list_inputs = []
        list_inputs.extend(model_inputs)
        list_inputs.append(1.)
    else:
        list_inputs = [model_inputs, 1.]

    # Learning phase. 1 = Test mode (no dropout or batch normalization)
    # layer_outputs = [func([model_inputs, 1.])[0] for func in funcs]
    layer_outputs = [func(list_inputs)[0] for func in funcs]
    for layer_activations in layer_outputs:
        activations.append(layer_activations)
        if print_shape_only:
            print(layer_activations.shape)
        else:
            print(layer_activations)
    return activations

# From:  https://github.com/philipperemy/keras-visualize-activations
def display_activations(activation_maps):
    import numpy as np
    import matplotlib.pyplot as plt
    """
    (1, 26, 26, 32)
    (1, 24, 24, 64)
    (1, 12, 12, 64)
    (1, 12, 12, 64)
    (1, 9216)
    (1, 128)
    (1, 128)
    (1, 10)
    """
    batch_size = activation_maps[0].shape[0]
    assert batch_size == 1, 'One image at a time to visualize.'
    for i, activation_map in enumerate(activation_maps):
        print('Displaying activation map {}'.format(i))
        shape = activation_map.shape
        print(len(shape))
        if len(shape) == 4:
            activations = np.hstack(np.transpose(activation_map[0], (2, 0, 1)))
        elif len(shape) == 2:
            # try to make it square as much as possible. we can skip some activations.
            activations = activation_map[0]
            num_activations = len(activations)
            if num_activations > 900:  # too hard to display it on the screen.
                square_param = int(np.floor(np.sqrt(num_activations)))
                activations = activations[0: square_param * square_param]
                activations = np.reshape(activations, (square_param, square_param))
            else:
                activations = np.expand_dims(activations, axis=0)
        else:
            raise Exception('len(shape) = 3 has not been implemented.')
        plt.imshow(activations, interpolation='None', cmap='jet')
        plt.show()

