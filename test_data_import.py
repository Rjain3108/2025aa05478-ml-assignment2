import kagglehub
from kagglehub import KaggleDatasetAdapter


# Set the path to the file you'd like to load
train_file_path = "train.csv"
test_file_path="test.csv"

def get_mobile_price_test():
    df = kagglehub.load_dataset( KaggleDatasetAdapter.PANDAS,"iabhishekofficial/mobile-price-classification",train_file_path)
    return df

def main():
    data = get_mobile_price_test()
    print(data.shape)

if __name__ == "__main__":
    main()

#df = kagglehub.load_dataset( KaggleDatasetAdapter.PANDAS,"iabhishekofficial/mobile-price-classification",train_file_path)
#print(df.shape)