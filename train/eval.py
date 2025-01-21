import numpy as np
import pandas as pd
import argparse

def rank_precision(data, k, id_column='hotel_id', predict_column='predict',label_column='label'):
    data.sort_values(predict_column,ascending=False,inplace=True)
    grouped=data.groupby(id_column)
    right=0
    total=0
    for _,group in grouped:
        labels=group[label_column].values[:k]
        if 1 in labels:
            right+=1
        total+=1
    precision=round(right/total,3)
    # print(f"total ids:{total}")
    print(f"rank@{k}:{precision}")
    return precision

def main(data_path):
    pred=pd.read_csv(data_path)
    pred['hotel_id']=[i.split('_')[0] for i in pred['image'].values]
    pred.sort_values('predict',ascending=False,inplace=True)

    rank_precision(pred,1)
    rank_precision(pred,2)
    rank_precision(pred,3)
    rank_precision(pred,4)
    rank_precision(pred,5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./pred_single_sample.csv')
    
    args = parser.parse_args()
    main(args.data_path)
