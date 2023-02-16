import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_bars(df, column):
    # Plot the ROC-AUC value from the non read_only dataset
    df = df[df["read_only"] == False]
    fig, ax = plt.subplots(figsize=(5, 4))
    # creating the bar plot
    plt.bar(df["trace_name"].tolist(), df[column].tolist(), color =[np.random.rand(3,) for x in range(len(df))], width = 0.4)
    plt.setp( ax.xaxis.get_majorticklabels(), rotation=30, ha="right" )
    # ax.yaxis.set_ticks(np.arange(0, 1.1, 0.2))
    plt.xlabel("Dataset name")
    plt.ylabel(column.upper())
    plt.title(column.upper() + " on Various Dataset")
    # plt.ylim(0,1)
    plt.show()
    plt.savefig(column+".png")

df = pd.read_csv("model_collection/1_per_io_admission/dataset/models_performance.csv")
df.head()

plot_bars(df, "roc_auc")
plot_bars(df, "fnr")
plot_bars(df, "fpr")