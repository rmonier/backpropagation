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

        # Calculate MAPE
        sum_abs_distance = 0
        sum_z = 0
        for i in range(len(y_test)):
            sum_abs_distance += abs(y_test[i] - y_pred[i])
            sum_z += y_pred[i]
        mape = 100 * (sum_abs_distance / sum_z)

        # Save the parameters and results in a CSV file
        df = pd.DataFrame({
            'Intercept': [mlr.intercept_],
            'Coefficients': [mlr.coef_],
            'MAPE': [mape]
        })
        df.to_csv(f"evaluation/results/mlr_{folder_path.name}.csv", encoding='utf8', index=False)
        
        print(f"Folder: {folder_path.name}\t| Accuracy: {mlr.score(x_test, y_test)*100} %")

if __name__ == '__main__':
    main()