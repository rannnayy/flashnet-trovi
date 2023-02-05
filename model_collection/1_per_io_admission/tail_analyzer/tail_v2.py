#!/usr/bin/env python3

import argparse
import csv
import numpy as np
import os
import sys
import subprocess
from subprocess import call
from pathlib import Path
import pandas as pd 
import math
import matplotlib.pyplot as plt

sys.path.append('../../../commonutils')
import default_ip_finder
import pattern_checker

def create_output_dir(output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    return output_path

# save to a file
def write_to_file(df, filePath, has_header=True):
    # The raw (replayed) traces don't have the header
    df.to_csv(filePath, index=False, header=has_header, sep=',')
    print("===== output file : " + filePath)

def write_stats(filePath, statistics):
    with open(filePath, "w") as text_file:
        text_file.write(statistics)
    print("===== output file : " + filePath)

def read_file(input_file):
    df = pd.read_csv(input_file, header=None, sep=',')
    # Make sure it has 7 columns
    assert 7 == df.shape[1]
    # Rename column
    # Format = ts_record(ms),latency(us),io_type(r=1/w=0),
    #          size(B),offset,ts_submit(ms),size_after_replay(B)
    df.columns = ["ts_record","latency","io_type","size","offset","ts_submit","size_after_replay"]

    # filter: remove io that doesn't executed properly (can't read/write all bytes)
    df = df[df['size'] == df['size_after_replay']]
    return df

def plot_raw_vs_best(figure_path, y_raw, y_best, extra_info=""):
    # Draw CDF
    N=len(y_best)
    data = y_best
    # sort the data in ascending order
    x_1 = np.sort(data)
    # get the cdf values of y
    y_1 = np.arange(N) / float(N)

    N=len(y_raw)
    data = y_raw
    # sort the data in ascending order
    x_2 = np.sort(data)
    # get the cdf values of y
    y_2 = np.arange(N) / float(N)

    # plotting
    plt.figure(figsize=(7,3))
    plt.xlabel('Latency (us)')
    plt.ylabel('CDF')
    plt.title('CDF of Latency (Read-only IOs) \n' + extra_info)
    p70_lat = np.percentile(y_raw, 70)
    plt.xlim(0, max(p70_lat * 3, 1000)) # Hopefully the x axis limit can catch the tail
    plt.ylim(0, 1) 
    plt.plot(x_2, y_2, label = "Raw latency", color="red")
    plt.plot(x_1, y_1, label = "FlashNet-best-case", color="green")
    plt.legend(loc="lower right")
    plt.savefig(figure_path, bbox_inches='tight')
    print("===== output figure : " + figure_path)

def calc_percent(partition, total, precision = 2):
    return str(round(partition*100/total,precision)) + "%"

def start_processing(input_path): 
# 1. Add more variable to Analyze the Trace
    df = read_file(input_path)
    stats_total_io = len(df)
    stats_n_read = len(df[df["io_type"] == 1])

    # Sort based on ts_submit, there is a slight out of order due to multithreading submission
    df = df.sort_values('ts_submit')
    df = df.reset_index(drop=True)

# 2. Find SLOW Latency based on IP Threshold 
    ip_latency_threshold, ip_latency_percent = default_ip_finder.tangent_based(df['latency'])
    
    if (ip_latency_percent < 50):
        print("ERROR: this trace profile is BAD because the IP latency is < 50%. Flashnet won't be able to make any significant improvement.")

    df['reject'] = df.apply (lambda row: 1 if row['latency'] > ip_latency_threshold else 0, axis=1)

# 3. Create Ouput dir
    stats_n_labeled = len(df)
    print("#IO labeled = " + str(stats_n_labeled))

    profile_name = os.path.basename(input_path)
    parent_dir_name = os.path.basename(Path(input_path).parent)
    profile_name = str(Path(profile_name).with_suffix('') ) # remove .trace extension
    output_dir = "../dataset/" + parent_dir_name + "/" + profile_name
    create_output_dir(output_dir)

# 8. Write data as labeled dataset
    # drop unnecessary columns 
    important_columns = ["ts_record", "io_type","size","offset","ts_submit","size_after_replay", "latency", "reject"]
    df = df.loc[:, df.columns.intersection(important_columns)]

    outfile_path = os.path.join(output_dir, "profile_v2.labeled")
    stats_n_fast_io = len(df[df['reject'] == 0])
    stats_n_slow_io = len(df[df['reject'] == 1])
    print("Fast IO = " + str(stats_n_fast_io))
    print("Slow IO = " + str(stats_n_slow_io))
    write_to_file(df, outfile_path, True)

# 9. Write the stats
    logging = []
    logging.append("============================================")
    logging.append("                BASIC INFO ")
    logging.append("============================================")
    logging.append("Profile name = " + profile_name)
    logging.append("Full path    = " + input_path)
    stats_read_ratio = int((stats_n_read/stats_total_io)*100)
    logging.append("R:W ratio    = " + str(stats_read_ratio) + ":" + str(100-stats_read_ratio))
    logging.append("#IO          = " + str(stats_total_io))
    logging.append("#writes      = " + str(stats_total_io - stats_n_read))
    logging.append("#reads       = " + str(stats_n_read))
    logging.append("============================================")
    logging.append("                STATISTICS")
    logging.append("============================================")
    logging.append("IP latency        = " + str(ip_latency_threshold) + " us ("+ str(round(ip_latency_percent, 2)) +"%)")
    logging.append("Median latency    = " + str(ip_latency_threshold) + " us (50%)")
    logging.append("Median throughput = " + str(ip_latency_threshold) + " B/us (50%)")
    logging.append("#IO labeled  = " + str(stats_n_labeled) + "  ("+calc_percent(stats_n_labeled,stats_total_io)+" out of " + str(stats_total_io) + ")")

    df = df[df["io_type"] == 1] # Only check the read
    stats_n_read_io_labeled = len(df)

    logging.append("  #Write IO  = " + str(stats_n_labeled - stats_n_read_io_labeled))
    logging.append("  #Read IO   = " + str(stats_n_read_io_labeled))

    logging.append("Fast R/W IO  = " + str(stats_n_fast_io) + "  ("+calc_percent(stats_n_fast_io,stats_n_labeled)+" out of " + str(stats_n_labeled) + ")")
    logging.append("Slow R/W IO  = " + str(stats_n_slow_io) + "  ("+calc_percent(stats_n_slow_io,stats_n_labeled)+" out of " + str(stats_n_labeled) + ")")
    
    stats_n_fast_read_io = len(df[df['reject'] == 0])
    stats_n_slow_read_io = len(df[df['reject'] == 1])
    stats_percent_fast_read = calc_percent(stats_n_fast_read_io,stats_n_read_io_labeled,0)
    stats_percent_slow_read = calc_percent(stats_n_slow_read_io,stats_n_read_io_labeled,0)
    logging.append("Fast Read-IO = " + str(stats_n_fast_read_io) + "  ("+stats_percent_fast_read+" out of " + str(stats_n_read_io_labeled) + ")")
    logging.append("Slow Read-IO = " + str(stats_n_slow_read_io) + "  ("+stats_percent_slow_read+" out of " + str(stats_n_read_io_labeled) + ")")
    outfile_path = os.path.join(output_dir, "profile_v2.stats")
    write_stats(outfile_path, "\n".join(logging))

# 10. Draw best-case CDF 
    figure_path = os.path.join(output_dir, "profile_v2.lat_cdf.png")
    y_best = df.loc[df['reject'] == 0, 'latency']
    y_raw = df['latency'].values
    extra_info = "[Outlier = None; Label proportion = " + stats_percent_fast_read + " fast and " + stats_percent_slow_read +  "slow ]"
    plot_raw_vs_best(figure_path, y_raw, y_best, extra_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dir", help="Directory path to find the trace profiles",type=str)
    parser.add_argument("-pattern", help="Pattern to match to the profile name",type=str)
    parser.add_argument("-file", help="File path of the trace profiles",type=str)
    parser.add_argument("-files", help="Arr of file path of the trace profiles", nargs='+',type=str)
    args = parser.parse_args()
    if (not args.file and not args.files and not (args.dir and args.pattern)):
        print("    ERROR: You must provide these arguments: -file <the input trace> ")
        exit(-1)

    trace_profiles = []
    if args.files:
        trace_profiles += args.files
    elif args.file:
        trace_profiles.append(args.file)
    print("trace_profiles = " + str(trace_profiles))
    
    for profile_path in trace_profiles:
        print("\nProcessing " + str(profile_path))
        start_processing(profile_path)
# How to run:
# ./tail_v1.py -file ../../data/trace_profile/nvme0n1/alibaba.cut.per_50k.most_thpt_size_iops_rand.141.trace