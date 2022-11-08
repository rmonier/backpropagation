import pandas as pd
import pathlib as pl

def main():
    for txt_file in pl.Path("datasets/preprocessed").glob('A1-*.txt'):
        df = pd.read_csv(f"datasets/preprocessed/{txt_file.name}", encoding='utf8', sep='	')
        df = (df - df.min()) / (df.max() - df.min())
        df.to_csv(f"datasets/normalized/{txt_file.name}", encoding='utf8', sep='	', index=False)

if __name__ == '__main__':
    main()
