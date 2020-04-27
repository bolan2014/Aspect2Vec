import pandas as pd
import json
import re
from collections import defaultdict
from src.utils.config import Configuration

pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.max_colwidth", 200)


def clean_mlc_aspects(aspt_str, return_dict=False):
    def mlc_data_check(aspt_str):
        aspt_list = aspt_str.lower().split(',')
        aspt = ','.join(i for i in aspt_list if i.find(':') > 0)
        aspt_list = aspt.split(':')
        aspt = ':'.join(i for idx, i in enumerate(aspt_list) if i.find(',') > 0 or idx == 0 or idx == len(aspt_list) - 1)
        return aspt

    aspt_str = aspt_str.replace('(', '').replace(')', '').replace('"', '')
    aspt = ('{"' + mlc_data_check(aspt_str) + '"}').replace(':', '":"').replace(',', '","')
    try:
        aspt_dict = json.loads(aspt)
    except json.JSONDecodeError:
        aspt_dict = defaultdict(lambda: 'None')
    if return_dict:
        return aspt_dict
    else:
        return '|'.join([f"{k}: {v}" for k, v in aspt_dict.items()])


def extract_flipkart_category(categ_str):
    root_categ_re = re.compile('([\&\sa-z,]+\s\>\>)')

    rs = root_categ_re.findall(categ_str.lower())
    if rs:
        return rs[0].replace('\"', '').replace(' >>', '')
    else:
        return categ_str.lstrip('[').rstrip(']').strip('\"')


def clean_flipkart_aspects(aspt_str):
    def flipkart_aspt_data_check(aspt_str):
        if not isinstance(aspt_str, str):
            return None
        aspt_list = aspt_str.lower().lstrip('{\"product_specification\"=>').rstrip('}]}').lstrip('[{').split('}, {')
        aspt = []
        for i in aspt_list:
            if i.find("key\"") >= 0 and i.find("value\"") >= 0:
                aspt.append(i)
        if aspt:
            return aspt
        else:
            return None

    aspt = flipkart_aspt_data_check(aspt_str)
    key_re = re.compile('key"=>"[^"]+"')
    value_re = re.compile('value"=>"[^"]+"')
    aspt_dict = {}
    if not aspt:
        return "None"
    for i in aspt:
        try:
            key = key_re.findall(i)[0].lstrip('key\"=>').strip('\"').strip()
            value = value_re.findall(i)[0].lstrip('value\"=>').strip('\"').strip()
            aspt_dict[key] = value
        except IndexError:
            print("lost key or value", i)
    return '|'.join([f"{k}: {v}" for k, v in aspt_dict.items()])



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    args = parser.parse_args()
    if args.data == 'mlc':
        config = Configuration('../../', suffix='ebay-mlc', file_name='validation_set.tsv')
        df = pd.read_csv(config.origin_file, sep='\t')
        df = df[["index", "leaf_categ_id", "attributes", "auct_titl", "cluster"]]
        df = df.rename(columns={"index": "uniq_id", "leaf_categ_id": "category", "auct_titl": "title", "attributes": "aspects"})
        print(df.columns)
        print(df.shape)
        df["aspects"] = df["aspects"].apply(lambda x: clean_mlc_aspects(x))
        df = df[df["aspects"] != '']
        print("dataset shape after cleaning the aspects: ", df.shape)

        df.sort_values(by="category", inplace=True)
        df.to_csv(config.cleaned_data_file, sep='\t', header=True, index=False)

    elif args.data == 'flipkart':
        config = Configuration('../../', suffix='flipkart', file_name='flipkart_com-ecommerce_sample.csv')
        df = pd.read_csv(config.origin_file, sep=',')
        df = df[["uniq_id", "product_category_tree", "product_specifications", "product_url", "brand", "product_name"]]
        df = df.rename(columns={"product_category_tree": "category", "product_specifications": "aspects", "product_name": "cluster"})
        print(df.columns)
        print(df.shape)
        df["category"] = df["category"].apply(lambda x: extract_flipkart_category(x))
        d = pd.DataFrame(df["category"].value_counts())

        categ_filter = [v for v, c in pd.DataFrame(df["category"].value_counts()).iterrows() if c[0] >= 100]
        df = df[df["category"].isin(categ_filter)]
        print(df["category"].value_counts())
        print("dataset shape after cleaning the category: ", df.shape)

        df["aspects"] = df["aspects"].apply(lambda x: clean_flipkart_aspects(x))
        df = df[df["aspects"] != "None"]
        print("dataset shape after cleaning the aspects: ", df.shape)

        df.sort_values(by="category", inplace=True)
        df.to_csv(config.cleaned_data_file, sep='\t', header=True, index=False)

