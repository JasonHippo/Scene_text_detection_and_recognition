import pandas as pd
import argparse

badword = ["!", "$","%","&","'","(",")","*","+",",","-",".","/",":",";","<","=",">","?","@","[","/","]","^","_","`","{","|","}","~","の","®"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--path', help='the path of file')
    opt = parser.parse_args()
    
    print(len(badword))
    df = pd.read_csv(opt.path)
    print(len(df))
    nan_list = list()
    for i in range(len(df)):
        row = df.loc[i]
        print(i)
        print(row)
        try:
            if any(bad_word in row['pred'] for bad_word in badword):
                ss = row['pred']
                for j in range(len(badword)):
                    ss = ss.replace(badword[j],"")
                if ss =="":
                    df.loc[i,'pred']= "の"
                    print(df.loc[i])  
                    continue  
                df.loc[i,'pred']= ss
        except:
            nan_list.append(i)
    df = df.drop(nan_list)
    df = df.drop(df.loc[df['pred']=='の'].index)
    
    print(df.head)
    df.to_csv("{}_post.csv".format(opt.path.split('.csv')[0]),index=False,encoding="utf-8")