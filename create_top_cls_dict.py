# coding=utf-8
import pandas as pd
import json


def main():
    df = pd.read_csv('/home/websensa/hdd/kaggle/data/category_names.csv',
                     header=0,
                     converters={
                         'category_id': str,
                         'category_level1': lambda x: x.encode('utf-8'),
                         'category_level2': lambda x: x.encode('utf-8'),
                         'category_level3': lambda x: x.encode('utf-8'),
                     },
                     encoding='utf-8'
                     )

    level1_cat_to_label = {}
    for cat in df['category_level1'].values:
        if cat not in level1_cat_to_label:
            level1_cat_to_label[cat] = len(level1_cat_to_label)

    level2_cat_to_label = {}
    for cat in df['category_level2'].values:
        if cat not in level2_cat_to_label:
            level2_cat_to_label[cat] = len(level2_cat_to_label)

    label_to_level1 = {label: cat for cat, label in level1_cat_to_label.items()}
    label_to_level2 = {label: cat for cat, label in level2_cat_to_label.items()}

    cat_to_label = {cat: i for i, cat in enumerate(df['category_id'].values)}
    label_to_cat = {i: cat for i, cat in cat_to_label.items()}

    cat_to_level2_label = {}
    for cat, lvl2 in zip(df['category_id'].values, df['category_level2'].values):
        cat_to_level2_label[cat] = level2_cat_to_label[lvl2]

    cat_to_level1_label = {}
    for cat, lvl1 in zip(df['category_id'].values, df['category_level1'].values):
        cat_to_level1_label[cat] = level1_cat_to_label[lvl1]

    with open('label_to_cat.json', 'w') as f:
        json.dump(label_to_cat, f)

    with open('cat_to_label.json', 'w') as f:
        json.dump(cat_to_label, f)

    with open('cat_to_level1_label.json', 'w') as f:
        json.dump(cat_to_level1_label, f)

    with open('cat_to_level2_label.json', 'w') as f:
        json.dump(cat_to_level2_label, f)


if __name__ == '__main__':
    main()
