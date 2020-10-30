import pandas as pd
import json
import re
from collections import defaultdict
from src.utils.config import Configuration

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
pd.set_option("display.max_colwidth", 200)


def clean_and_dump_dataset(config, dump_flag=True):
    if config.data_tag == 'mlc':
        df = pd.read_csv(config.origin_file, sep='\t')
        df = df[["index", "leaf_categ_id", "attributes", "auct_titl", "cluster"]]
        df = df.rename(
            columns={"index": "uniq_id", "leaf_categ_id": "category", "auct_titl": "title", "attributes": "aspects"})
        print(df.columns)
        config.logging.info(f"Raw data shaped in {df.shape}")
        df["aspects"] = df["aspects"].apply(lambda x: clean_mlc_aspects(x))
        df = df[df["aspects"] != "None"]
        config.logging.info(f"Cleaned data shaped in{df.shape}")

        df.sort_values(by="category", inplace=True)
        if dump_flag:
            df.to_csv(config.cleaned_data_file, sep='\t', header=True, index=False)
    elif config.data_tag == 'flipkart':
        df = pd.read_csv(config.origin_file, sep=',')
        df = df[["uniq_id", "product_category_tree", "product_specifications", "product_url", "brand", "product_name"]]
        df = df.rename(columns={"product_category_tree": "category", "product_specifications": "aspects",
                                "product_name": "cluster"})
        print(df.columns)
        config.logging.info(f"Raw data shaped in {df.shape}")
        df["category"] = df["category"].apply(lambda x: extract_flipkart_category(x))

        categ_filter = [v for v, c in pd.DataFrame(df["category"].value_counts()).iterrows() if c[0] >= 100]
        df = df[df["category"].isin(categ_filter)]
        print(df["category"].value_counts())
        config.logging.info(f"Dataset shape after cleaning the category: {df.shape}")

        df["aspects"] = df.apply(lambda x: clean_flipkart_aspects(x["aspects"], x["brand"]), axis=1)
        df = df[df["aspects"] != "None"]
        config.logging.info(f"Dataset shape after cleaning the aspects: {df.shape}")

        df.sort_values(by="category", inplace=True)
        if dump_flag:
            df.to_csv(config.cleaned_data_file, sep='\t', header=True, index=False)
    else:
        df = None
        config.logging.error("Data not loaded")
    return df

def drop_nonsense_aspect(aspt_dict):
    _aspt_dict = {}
    for k, v in aspt_dict.items():
        if v.find("unknown") >= 0 \
                or v.find("does not apply") >= 0 \
                or v.find("unbranded") >= 0 \
                or v.find("no") >= 0 \
                or v.find("none") >= 0 \
                or v.find("na") >= 0 \
                or k.find("other details") >= 0:
            pass
        else:
            _aspt_dict[k] = v

    return _aspt_dict


def clean_mlc_aspects(aspt_str, return_dict=False):
    def mlc_data_check(aspt_str):
        aspt_list = aspt_str.lower().split(',')
        aspt = ','.join(i for i in aspt_list if i.find(':') > 0)
        aspt_list = aspt.split(':')
        aspt = ':'.join(
            i for idx, i in enumerate(aspt_list) if i.find(',') > 0 or idx == 0 or idx == len(aspt_list) - 1)
        return aspt

    aspt_str = aspt_str.replace('(', '').replace(')', '').replace('"', '')
    aspt = ('{"' + mlc_data_check(aspt_str) + '"}').replace(':', '":"').replace(',', '","')
    try:
        aspt_dict = json.loads(aspt)
    except json.JSONDecodeError:
        aspt_dict = defaultdict(lambda: 'None')
    aspt_dict = drop_nonsense_aspect(aspt_dict)
    if return_dict:
        return aspt_dict
    elif aspt_dict:
        return '|'.join([f"{k}: {v}" for k, v in aspt_dict.items()])
    else:
        return "None"


def extract_flipkart_category(categ_str):
    root_categ_re = re.compile('([\&\sa-z,]+\s\>\>)')

    rs = root_categ_re.findall(categ_str.lower())
    if rs:
        return rs[0].replace('\"', '').replace(' >>', '')
    else:
        return categ_str.lstrip('[').rstrip(']').strip('\"')


def clean_flipkart_aspects(aspt_str, brand_str=None, return_dict=False):
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
    if isinstance(brand_str, str):
        aspt_dict["brand"] = brand_str.lower().strip()
    aspt_dict = drop_nonsense_aspect(aspt_dict)
    if return_dict:
        return aspt_dict
    elif aspt_dict:
        return '|'.join([f"{k}: {v}" for k, v in aspt_dict.items()])
    else:
        return "None"


if __name__ == '__main__':
    cf = Configuration('../../', suffix='flipkart', file_name='flipkart_com-ecommerce_sample.csv')
    # cf = Configuration('../../', suffix='ebay', file_name='human_labeled_set.tsv')
    clean_and_dump_dataset(config=cf)
