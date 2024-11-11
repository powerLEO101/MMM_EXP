# --- 

import sys
sys.path.append('/media/workspace/MMM_EXP')

# --- Start
import metrics
import argparse
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate a CSV file')
    parser.add_argument('--csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('--solution_file', type=str, help='Path to the solution CSV file')
    parser.add_argument('--misconceptions', type=str, help='Mis')
    parser.add_argument('--fold', type=int, default=0, help='Fold number')
    return parser.parse_args()

def fake_arg_parser():
    return argparse.Namespace(csv_file='submission.csv', solution_file='train.csv', misconceptions='misconception_mapping.csv', fold=4)

args = parse_args()

# Read the CSV file
prediction = pd.read_csv(args.csv_file)
train_df = pd.read_csv(args.solution_file)
misconceptions = pd.read_csv(args.misconceptions)

train_df['fold'] = train_df['QuestionId'].apply(lambda x: x % 5)
train_df = train_df[train_df['fold'] == args.fold]

solution = []
for idx, row in train_df.iterrows():
    for choice in ['A', 'B', 'C', 'D']:
        if choice == row['CorrectAnswer']: continue
        if np.isnan(row[f'Misconception{choice}Id']): continue
        solution.append([f"{row['QuestionId']}_{choice}", int(row[f'Misconception{choice}Id'])])
solution = pd.DataFrame(solution, columns=['QuestionId_Answer', 'MisconceptionId'])

# evaluate using metrics.mapk function

actual, predicted = [], []
for idx, row in solution.iterrows():
    if row['QuestionId_Answer'] not in prediction['QuestionId_Answer'].values:
        predicted.append([])
    else:
        predicted.append([int(x) for x in prediction[prediction['QuestionId_Answer'] == row['QuestionId_Answer']]['MisconceptionId'].values[0].split()])
    actual.append([row['MisconceptionId']])
mapk_score = metrics.mapk(actual, predicted, k=25)
print('MAP@25:', mapk_score)
# --- End 
