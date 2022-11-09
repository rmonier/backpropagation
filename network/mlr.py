from sklearn.linear_model import LinearRegression
import pandas as pd
import pathlib as pl

def main():
    for folder_path in pl.Path("datasets/split").glob('A1-*'):
        df_train = pd.read_csv(f"datasets/split/{folder_path.name}/train.txt", encoding='utf8', sep='	')
        df_test = pd.read_csv(f"datasets/split/{folder_path.name}/test.txt", encoding='utf8', sep='	')

        x_train = df_train.iloc[:, :-1]
        y_train = df_train.iloc[:, -1]
        x_test = df_test.iloc[:, :-1]
        y_test = df_test.iloc[:, -1]

        mlr = LinearRegression()  
        mlr.fit(x_train, y_train)

        y_pred = mlr.predict(x_test)
        print(f"Folder: {folder_path.name}\t| Accuracy: {mlr.score(x_test, y_test)*100} %")

if __name__ == '__main__':
    main()