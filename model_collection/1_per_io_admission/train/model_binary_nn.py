#!/usr/bin/env python3

import argparse
import numpy as np
import os
import sys
from subprocess import call
from pathlib import Path
import pandas as pd 
import matplotlib.pyplot as plt
import PIL
from PIL import Image

import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, roc_auc_score, classification_report
import matplotlib.pyplot as plt

def create_output_dir(output_path):
    os.makedirs(output_path, exist_ok=True)
    return output_path

def write_stats(filePath, statistics):
    with open(filePath, "w") as text_file:
        text_file.write(statistics)
    print("===== output file : " + filePath)

def print_confusion_matrix(figure_path, y_test, y_pred):
    y_test_class, y_pred_class = y_test, y_pred
    target_names = ["Fast", "Slow"]
    labels_names = [0,1]
    stats = []
    stats.append(classification_report(y_test_class, y_pred_class,labels=labels_names, target_names=target_names))

    fig, ax = plt.subplots(figsize=(4, 3))

    cm = confusion_matrix(y_test_class, y_pred_class)

    # Calculate FNR
    TP = cm[1][1]
    TN = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    FPR, FNR = round(FP/(FP+TN),3), round(FN/(TP+FN),3)
    ROC_AUC = round(roc_auc_score(y_test_class, y_pred_class),3)
    stats.append("FPR = "+ str(FPR) + "  (" + str(round(FPR*100,1))+ "%)")
    stats.append("FNR = "+ str(FNR) + "  (" + str(round(FNR*100,1))+ "%)")
    stats.append("ROC-AUC = "+ str(ROC_AUC) + "  (" + str(round(ROC_AUC*100,1))+ "%)")
    
    disp = ConfusionMatrixDisplay(cm, display_labels=target_names)
    disp = disp.plot(ax=ax, cmap=plt.cm.Blues,values_format='g')
    ax.set_title("FPR = " + str(round(FPR*100,1))+ "%  and FNR = " + str(round(FNR*100,1))+ "%"); 

    # FN -> bottom left corner
    plt.savefig(figure_path, bbox_inches='tight')
    # print("===== output figure : " + figure_path)
    return stats, ROC_AUC

def plot_latency_cdf(figure_path, complete_df, title):
    # the df is already readonly IOs
    y_pred = complete_df.loc[complete_df["y_pred"] == 0, "latency"].values
    # Draw CDF
    N=len(y_pred)
    data = y_pred
    # sort the data in ascending order
    x_1 = np.sort(data)
    # get the cdf values of y
    y_1 = np.arange(N) / float(N)

    y_test = complete_df["latency"].values
    N=len(y_test)
    data = y_test
    # sort the data in ascending order
    x_2 = np.sort(data)
    # get the cdf values of y
    y_2 = np.arange(N) / float(N)

    # plotting
    plt.figure(figsize=(6,3))
    plt.xlabel('Latency (us)')
    plt.ylabel('CDF')
    plt.title(title)
    p70_lat = np.percentile(x_2, 70)
    plt.xlim(0, max(p70_lat * 3, 1000)) # Hopefully the x axis limit can catch the tail
    plt.ylim(0, 1) 
    plt.plot(x_2, y_2, label = "Raw Latency", color="red")
    plt.plot(x_1, y_1, label = "FlashNet-powered", linestyle='dashdot', color="green")
    plt.legend(loc="lower right")
    plt.savefig(figure_path, bbox_inches='tight')
    # print("===== output figure : " + figure_path)

def plot_loss(figure_path, history):
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    # plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
    plt.savefig(figure_path, bbox_inches='tight')

def train_model(dataset_path, percent_data_for_eval):
    dataset = pd.read_csv(dataset_path)
    
# Put "latency" at the end
    reordered_cols = [col for col in dataset.columns if col != "latency"] + ["latency"]
    dataset = dataset[reordered_cols]

# Split test and training set
    x = dataset.copy(deep=True).drop(columns=["reject"], axis=1)
    y = dataset["reject"].copy(deep=True)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=percent_data_for_eval/100, random_state=42)

# Remove the latency column from the input feature
    # remove latency data from X_train and X_test
    x_train_latency = x_train['latency']
    x_test_latency = x_test['latency']
    x_train = x_train.drop(columns=["latency"], axis=1) # Avoid using current latency IO as the input feature
    x_test = x_test.drop(columns=["latency"], axis=1)

# Data normalization
    normalizer = layers.Normalization(axis=-1)
    normalizer.adapt(np.array(x_train))
    print(normalizer.mean.numpy())

# Model Architecture
    stats = []
    dnn_model = keras.Sequential([
        normalizer,
        layers.Dense(512, activation='relu', input_dim=x_train.shape[1]),
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    dnn_model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.0001), metrics=['accuracy'])
    model_summ = []
    dnn_model.summary(print_fn=lambda x: model_summ.append(x))
    stats += ["\n".join(model_summ)]

# Output Directory
    dataset_name = str(Path(os.path.basename(dataset_path)).with_suffix(''))
    model_name = str(Path(os.path.basename(__file__)).with_suffix(''))
    parent_dir_name = Path(dataset_path).parent
    output_dir = os.path.join(parent_dir_name, dataset_name , model_name)   
    create_output_dir(output_dir)

# Train the model
    history = dnn_model.fit(
        x_train,
        y_train,
        validation_split=0.2,
        verbose=1, epochs=10
    )
    figure_path0 = os.path.join(output_dir, "train_loss.png")
    plot_loss(figure_path0, history)

# Evaluation
    y_pred = (dnn_model.predict(x_test) > 0.5).flatten()
    # print(y_pred[:10])
    
# Add the latency back to the X_test to draw the CDF Latency
    # must be done after the evaluation
    x_test['latency'] = x_test_latency

# Print confusion matrix and stats
    figure_path1 = os.path.join(output_dir, "conf_matrix.png")
    conf_matrix_stats, ROC_AUC = print_confusion_matrix(figure_path1, y_test, y_pred)
    stats += conf_matrix_stats
    
    outfile_path = os.path.join(output_dir, "eval.stats")
    write_stats(outfile_path, "\n".join(stats))

# Plot CDF Figures
    # original column names without the "reject" column
    columns = dataset.columns.to_list()
    columns.remove('reject')

    # Construct the chosen test set in a dataframe
    X_test_df = x_test.copy(deep=True)
    X_test_df["y_test"] = y_test # Real/True decision
    X_test_df["y_pred"] = y_pred
    # Keep the read IO only
    X_test_df = X_test_df[X_test_df['io_type'] == 1]
    figure_path2 = os.path.join(output_dir, "flashnet_cdf.png")
    title = "Read-IO Latency CDF [ROC-AUC = "+ str(ROC_AUC) + " = " + str(round(ROC_AUC*100,1))+ "%] \n model = " + model_name + " ;  eval = " + str(percent_data_for_eval) + "%"
    plot_latency_cdf(figure_path2, X_test_df, title)

# Combine all figures
    list_im = [figure_path2, figure_path1]
    images    = [ Image.open(i) for i in list_im ]
    widths, heights = zip(*(i.size for i in images))
    # https://stackoverflow.com/questions/30227466/
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    figure_path_final = os.path.join(output_dir, "eval.png")
    new_im.save(figure_path_final)
    print("===== output figure : " + figure_path_final)

    # Delete figures after we combine them into a single figure
    os.remove(figure_path1)
    os.remove(figure_path2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset", help="Path to the dataset", type=str)
    parser.add_argument("-train_eval_split", help="Ratio to split the dataset for training and evaluation",type=str)
    args = parser.parse_args()
    if (not args.dataset and not args.train_eval_split):
        print("    ERROR: You must provide these arguments: -dataset <the labeled trace>  -train_eval_split <the split ratio> ")
        exit(-1)

    # print("Dataset " + args.dataset)
    ratios = args.train_eval_split.split("_")
    percent_data_for_training = int(ratios[0])
    percent_data_for_eval = int(ratios[1])
    assert( percent_data_for_training + percent_data_for_eval == 100)
    train_model(args.dataset, percent_data_for_eval)
    