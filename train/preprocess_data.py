import pandas as pd
#pip install openpyxl
from tqdm import tqdm
import json
import argparse
import os
import random
import requests
import shutil


def download_image(url, filename):
    """Downloads an image from a URL and saves it to a file.

    Args:
    url: The URL of the image.
    filename: The filename to save the image as.
    """
    response = requests.get(url)
    # Check for successful response status code
    if response.status_code == 200:
        with open(filename, 'wb') as f:
            f.write(response.content)
        # print(f"Image downloaded successfully: {filename}")
    else:
        print(f"Failed to download image: {url}")
        return 0
    return 1


def main(first_page_data_path, total_data_path, category_data_path, negative_sample_number_per_category, output_folder):
    filter_class = ['Food and Dining', 'Transportation', 'Staircase/Elevator', 'Other', 'Fitness Facility']

    first_page = pd.read_excel(first_page_data_path, sheet_name='收集')

    data1 = pd.read_excel(total_data_path, sheet_name='Sheet0')
    data2 = pd.read_excel(total_data_path, sheet_name='Sheet1')
    total = pd.concat([data1, data2])
    total.drop_duplicates('img_url', inplace=True)

    # select negative samples ： ## 按预测类别选择一个负样本备选集合，减少图片下载量
    pic_cates = pd.read_csv(category_data_path, header=None)
    pic_cates.columns = ['img_url', 'cate']
    for c in filter_class:
        pic_cates = pic_cates[~pic_cates['cate'].str.contains(c)]
    total = pd.merge(total, pic_cates, on=['img_url'], how='inner')
    print('total effective negtive samples:', total.shape[0])

    cates_num_select = {}
    select_negtive_pics = {}
    for (id, url, cate) in total.values:
        cate = cate.split('<')[0]
        num = cates_num_select.get(cate, 0)
        if num >= negative_sample_number_per_category * 5:  # 先筛选，后面过滤
            continue
        cates_num_select[cate] = num + 1
        select_negtive_pics[url] = id
    total = total[total['img_url'].isin(list(select_negtive_pics.keys()))]
    print('to download negative samples:', total.shape[0])

    '''
    # download first page pics
    os.makedirs(os.path.join(output_folder, 'first_page_pics'), exist_ok=True)
    for i in tqdm(range(first_page.shape[0])):
        # Example usage
        image_url = first_page['首图图片url'].values[i]
        id = first_page['酒店id'].values[i]
        filename = str(id) + '_' + image_url.split('/')[-1].split('.')[0] + '.jpg'
        download_image(image_url, os.path.join(output_folder, 'first_page_pics', filename))

    # download other pics
    os.makedirs(os.path.join(output_folder, 'imgs'), exist_ok=True)
    for image_url, id in select_negtive_pics.items():
        filename = str(id) + '_' + image_url.split('/')[-1].split('.')[0] + '.jpg'
        try:
            download_image(image_url, os.path.join(output_folder, 'imgs', filename))
        except Exception as e:
            print(e)
            continue
    shutil.copytree(os.path.join(output_folder, 'first_page_pics'), os.path.join(output_folder, 'imgs'),
                    dirs_exist_ok=True)
    print("<<< download excel data!")
    '''

    # negative samples
    downloads = set(os.listdir(os.path.join(output_folder, 'imgs')))
    first_page_downloads = set(os.listdir(os.path.join(output_folder, 'first_page_pics')))
    neg_pics = {}
    cates_num_select = {}
    filename_set=set()
    for (id, url, cate) in total.values:
        id = str(id)
        cate = cate.split('<')[0]
        filename = str(id) + '_' + url.split('/')[-1].split('.')[0] + '.jpg'
        if filename in first_page_downloads or filename in filename_set:
            continue
        if filename not in downloads:
            continue
        num = cates_num_select.get(cate, 0)
        if num >= negative_sample_number_per_category:
            continue
        cates_num_select[cate] = num + 1
        filename_set.add(filename)
        ll = neg_pics.get(id, [])
        ll.append(filename)
        neg_pics[id] = ll
    # print(cates_num_select)
    # print(len(neg_pics))
    
    # train & test data
    pos_pics = {i.split('_')[0]: i for i in first_page_downloads}
    hotel_ids = [i for i in pos_pics.keys() if i in neg_pics]
    test_ids = random.sample(hotel_ids, 30)
    train = []
    test = []
    infer = []
    for id in pos_pics:
        if id not in neg_pics:
            # print(id)
            continue
        # print(len(neg_pics[id]))
        for img in neg_pics[id]:
            if random.sample([1, 2], 1)[0] == 1:
                sample = {"image1": pos_pics[id], "image2": img, "label": 1}
            else:
                sample = {"image1": img, "image2": pos_pics[id], "label": 0}
            if id in test_ids:
                test.append(sample)
                infer.extend([{"image": pos_pics[id], "label": 1},{"image": img, "label": 0}])
            else:
                train.append(sample)
    infer = [dict(t) for t in {tuple(d.items()) for d in infer}]
    train = [dict(t) for t in {tuple(d.items()) for d in train}]
    test = [dict(t) for t in {tuple(d.items()) for d in test}]
    print(f"train samples:{len(train)},test samples:{len(test)},infer samples:{len(infer)}")

    # Open a file for writing (text mode with UTF-8 encoding)
    with open(os.path.join(output_folder, 'train.json'), 'w') as f:
        json.dump(train, f)  # Optional parameter for indentation
    print('Data written to json')

    with open(os.path.join(output_folder, 'test.json'), 'w') as f:
        json.dump(test, f)  # Optional parameter for indentation
    print('Data written to json')
    
    with open(os.path.join(output_folder, 'infer.json'), 'w') as f:
        json.dump(infer, f)  # Optional parameter for indentation
    print('Data written to json')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--first_page_data_path', type=str, default='../klook_data/酒店首图.xlsx')
    parser.add_argument('--total_data_path', type=str, default='../klook_data/酒店全量图片.xls')
    parser.add_argument('--category_data_path', type=str, default='../klook_data/category_predict.txt')
    parser.add_argument('--negative_sample_number_per_category', type=int, default=100)
    parser.add_argument('--output_folder', type=str, default='../klook_data')
    args = parser.parse_args()
    main(args.first_page_data_path, args.total_data_path, args.category_data_path,
         args.negative_sample_number_per_category, args.output_folder)
