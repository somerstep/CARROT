import pickle
import re
from fuzzywuzzy import fuzz
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm
import copy
import matplotlib.pyplot as plt
from datetime import datetime
import os

def clone_repo(repo_url, target_directory='raw_data'):
    command = f'git clone {repo_url} {target_directory}'
    exit_code = os.system(command)
    if exit_code == 0:
        print(f"Repository cloned into {target_directory}")
        return f"Repo {repo_url}. Repository cloned into {target_directory}"
    else:
        print(f"An error occurred. Exit code: {exit_code}")
        return f"Repo {repo_url}. An error occurred. Exit code: {exit_code}"
    
def filter(s):
    try:s = s.split("/")[1]
    except: s = s
    try:s = s.split("__")[1]
    except: s = s
    return s.lower().replace("-hf","").replace("_","").replace("-","")
    
def search(s, s_list):
    scores = [fuzz.token_sort_ratio(filter(s), filter(s_try)) for s_try in s_list]
    return [s_list[np.argmax(scores)], np.max(scores)]

def transform_subscenarios_old(subscenarios):
    transformed = [
        re.sub(r'_(\d+)', r'|\1', scenario.replace('harness_', '')
                                              .replace('hendrycksTest_', 'hendrycksTest-')
                                              .replace('arc_challenge', 'arc:challenge')
                                              .replace('truthfulqa_mc', 'truthfulqa:mc'))
        for scenario in subscenarios
    ]
    return transformed
    
def find_folder_with_file_old(scenario, base_dir):
    # Get all folder names in the base directory
    folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
    folders = [f for f in folders if len(f)>=8]
    
    # Filter out folders that do not start with a date
    date_folders = [f for f in folders if f[4] == '-' and f[7] == '-']
    
    # Sort folders by date in descending order
    date_folders.sort(key=lambda date: datetime.strptime(date[:10], "%Y-%m-%d"), reverse=True)
    
    # Loop through the sorted folders
    for folder in date_folders:
        folder_path = os.path.join(base_dir, folder)
        # Check for files containing "x|5_" in their name
        for file in os.listdir(folder_path):
            if f"{scenario}_" in file:
                return folder


def find_folder_with_file_new(scenario, base_dir):
    files = [f for f in os.listdir(base_dir)]

    # Sort folders by date in descending order
    files = [f for f in files if re.findall(r"\d{4}-\d{2}-\d{2}T", f)!=[]]
    files.sort(key=lambda date: datetime.strptime(re.findall(r"\d{4}-\d{2}-\d{2}T", date)[0][:-1], "%Y-%m-%d"), reverse=True)
    
    
    # Loop through the sorted folders
    for file in files:
        file_path = os.path.join(base_dir, file)
        # Check for files containing "x|5_" in their name
        if f"{scenario}_" in file:
            return(file)
    
    return None