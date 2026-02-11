import pandas as pd
import kagglehub
from kagglehub import KaggleDatasetAdapter
from sklearn.model_selection import train_test_split


train_file_path = "train.csv"
test_file_path="test.csv"
# Read the dataset file (example)
#df = kagglehub.load_dataset( KaggleDatasetAdapter.PANDAS,"iabhishekofficial/mobile-price-classification",train_file_path)
df = kagglehub.load_dataset( KaggleDatasetAdapter.PANDAS,"iabhishekofficial/mobile-price-classification",test_file_path)

#train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
# Do any preprocessing if needed


# Save to new CSV
#train_df.to_csv("train.csv", index=False)
#test_df.to_csv("test.csv", index=False)
df= df.drop('id', axis=1)
print(df.head()) 
df.to_csv("prediction.csv", index=False)
