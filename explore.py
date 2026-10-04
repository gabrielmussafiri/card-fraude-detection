import pandas as pd
import numpy as np


df = pd.read_csv('./data/creditcard.csv')

print('=== SHAPE ===')
print(df.shape)

print('===Class Distribution ===')
print(df['Class'].value_counts())

print('=== Class Distribution (%) ===')
print(df['Class'].value_counts(normalize=True) * 100)

print('=== Missing Values ===')
print(f'Missing values: {df.isnull().sum().sum()}')

print("\n=== AMOUNT STATS BY CLASS ===")
print(df.groupby('Class')['Amount'].describe())

print("\n=== TOP 10 NEGATIVE CORRELATIONS WITH CLASS ===")
correlations = df.corr()['Class'].sort_values()
print(correlations.head(10))

print("\n=== TOP 10 POSITIVE CORRELATIONS WITH CLASS ===")
print(correlations.tail(10))
