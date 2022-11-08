import pandas as pd

#TODO: fix the filtering (0 line)
def main():
    df = pd.read_csv("datasets/raw/A1-top10s.csv")
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    del df[df.columns[0]]
    df.drop(['top genre', 'title', 'artist', 'year'], axis=1, inplace=True)
    df = df.astype(float)
    
    for i in range(len(df.columns)):
        data_mean, data_std = df.iloc[:,i].mean(), df.iloc[:,i].std()
        #with try and error -> 5 seems to be a good value
        cut_off = data_std * 5 
        lower, upper = data_mean - cut_off, data_mean + cut_off
        if i != 3 and i !=7 and i != 9:
            lower = 0
        df=df[(df.iloc[:,i] >= lower) & (df.iloc[:,i] <= upper)]
    
    df.rename(columns={"bpm":"#bpm"}, inplace=True)  # dirty hack but it works to have the desired #
    df.to_csv("datasets/preprocessed/A1-top10s.txt", encoding='utf8', sep='	', index=False)

if __name__ == '__main__':
    main()
