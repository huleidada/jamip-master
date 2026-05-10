import pandas as pd

def shellform(dataframe):

    #variables = list(data[0].keys())
    #dataframe = pd.DataFrame([[i[j] for j in variables] for i in data], columns=variables)
    #print(dataframe)
    
    info = dataframe.columns
    df = dataframe.fillna('--').astype(str)
    df_col_len = []
    for col in info:
        col_len = len(col)
        try:
            df_len = int(df[col].str.len().max())
        except:
            df_len = 10
        df_col_len.append(max(col_len, df_len)+2)

    def printGroup(row):
        for i,item in enumerate(row):
            if item == '-':
                s = str(item).center(df_col_len[i], '-')
                icon = '+'
            else:
                s = str(item).center(df_col_len[i], ' ')
                icon = '|'
    
            s = (icon if i == 0 else '') + s + icon
            print(s,end='')
        print('')
    
    print('\033[0;32;40mJAMIP Tasks Check Mode\033[0m')

    tag = ['-'] * len(info)
    printGroup(tag)
    printGroup(info)
    printGroup(tag)
    for row in df.itertuples(index=False):
        printGroup(row)
    printGroup(tag)
