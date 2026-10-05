import pandas as pd
import json


df = pd.read_csv('./data/creditcard.csv')

fraud_sample =df[df['Class'] == 1].iloc[0].drop('Class').to_dict()
normal_sample = df[df['Class'] ==0].iloc[0].drop('Class').to_dict()

print("FRAUD:")
print(json.dumps(fraud_sample, indent=2))
print("\nNORMAL:")
print(json.dumps(normal_sample, indent=2))