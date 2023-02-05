#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
import argparse

sys.path.append('../../../commonutils')
import pattern_checker

# save to a file
def write_to_file(df, filePath, has_header=True):
    df.to_csv(filePath, index=False, header=has_header, sep=',')
    print("===== output file : " + filePath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-files", help="Arr of file path of the evaluation statistics", nargs='+',type=str)
    args = parser.parse_args()
    if (not args.files):
        print("    ERROR: You must provide these arguments: -files <the statistics files> ")
        exit(-1)
    
    eval_files = []
    if args.files:
        eval_files += args.files
    print("Found " + str(len(eval_files)) + " stats files")

    df = pd.DataFrame(columns=["model_name", "device", "trace_name", "profile_ver", "feat_ver", "read_only", "fpr", "fnr", "roc_auc"])

    for file_path in eval_files:
        dirs = file_path.split("/")
        dirs.pop() # eval.stats
        model_name = dirs.pop()
        training_set = dirs.pop() # profile_v1.feat_v1.readonly
        read_only = True if "readonly" in training_set else False
        profile_feature = training_set.split(".") 
        profile_ver = profile_feature[0].split('_')[1]
        feat_ver = profile_feature[1].split('_')[1]
        trace_name=dirs.pop()
        device=dirs.pop()
        with open(file_path) as f:
            fpr = -1
            fnr = -1
            roc_auc = -1
            for line in f:
                # FPR = 0.044  (4.4%)
                # FNR = 0.239  (23.9%)
                # ROC-AUC = 0.858  (85.8%)
                if "FPR" in line:
                    fpr = float(line.split(" ")[2])
                elif "FNR" in line:
                    fnr = float(line.split(" ")[2])
                elif "ROC-AUC" in line:
                    roc_auc = float(line.split(" ")[2])
        df.loc[len(df)] = [model_name, device, trace_name, profile_ver, feat_ver, read_only, fpr, fnr, roc_auc]
        # print(df)
        # break
    write_to_file(df, "../dataset/models_performance.csv", True)
        