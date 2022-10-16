import pandas as pd

def main():
    df = pd.read_csv("datasets/raw/A1-top10s.csv")
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    del df[df.columns[0]]
    df.drop(['top genre', 'title', 'artist', 'year'], axis=1, inplace=True)
    df = df.astype(float)
    df.rename(columns={"bpm":"#bpm"}, inplace=True)  # dirty hack but it works to have the desired #
    df.to_csv("datasets/preprocessed/A1-top10s.txt", encoding = 'utf8', sep='	', index=False)

if __name__ == '__main__':
    main()

