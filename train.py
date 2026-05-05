import pandas as pd
import argparse
import os
import torch
import wandb
from sklearn.model_selection import train_test_split

from model.KNN import train_KNN
from model.decision_tree import X_train, train_decision_tree
from model.MLP import train_MLP

def main(args):
    parse = argparse.ArgumentParser(description="train model classification dog breed")
    parse.add_argument('--model', type=str, default='knn',help='choose model to train: knn, decision_tree, mlp')
    parse.add_argument('--data_path', type=str, default='dog_dataset.csv', help='path to preprocessed dataset')
    parse.add_argument('--test_size', type=float, default=0.2, help='proportion of test set')
    parse.add_argument('--random_state', type=int, default=42, help='random state for train-test split')
    parse.add_argument('--lr', type=float, default=0.1, help='learning rate for MLP')
    parse.add_argument('--epochs', type=int, default=10000, help='number of epochs for MLP')
    parse.add_argument('--hidden_size', type=int, default=8, help='number of neurons in hidden layer for MLP')
    parse.add_argument('--dropout', type=float, default=0.5, help='dropout rate for MLP')
    parse.add_argument('--max_depth', type=int, default=3, help='maximum depth for decision tree')
    parse.add_argument('--k', type=int, default=3, help='number of neighbors for KNN')

    args = parse.parse_args(args)
    if not os.path.exists(args.data_path):
        print(f"Data file {args.data_path} not found.")
        return
    print(f"Loading data from {args.data_path}...")
    
    if args.model == 'mlp':
        name = f"{args.model}_train_hidden{args.hidden_size}_lr{args.lr}_epochs{args.epochs}_dropout{args.dropout}"
    elif args.model == 'decision_tree':
        name = f"{args.model}_train_maxdepth{args.max_depth}"
    elif args.model == 'knn':
        name = f"{args.model}_train_k{args.k}"
    wandb.init(
        project='dog_breed_classification', 
        config=vars(args),
        name = f"{args.model}_train"
        )
    
    #check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    df = pd.read_csv(args.data_path)
    X = df.drop('breed', axis=1).values
    y = df['breed'].values
    X.train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=args.random_state)
    if args.model == 'knn':
        print("Training KNN model...")
        train_KNN(X_train, y_train, X_test, y_test, k=args.k)
    if args.model == 'decision_tree':
        print("Training Decision Tree model...")
        train_decision_tree(X_train, y_train, X_test, y_test, max_depth=args.max_depth)
    if args.model == 'mlp':
        print("Training MLP model...")
        train_MLP(X_train, y_train, X_test, y_test, hidden_size=args.hidden_size, lr=args.lr, epochs=args.epochs, dropout_rate=args.dropout)
    wandb.finish()