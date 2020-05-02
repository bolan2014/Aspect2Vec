import pandas as pd
import logging
from tqdm import tqdm
from src.utils.config import Configuration


class PairsetGenerator(object):
    def __init__(self, df, category: list, negative_ratio=5):
        self.origin_df = df[df["category"].isin(category)].reset_index()
        if len(self.origin_df) == 0:
            logging.error("no such categories in the dataframe")
        self.origin_df["uniq_id"] = self.origin_df["uniq_id"].astype(str)
        self.negative_ratio=negative_ratio
        self.category = category
        self.total_pos = 2000

    def make_pairs(self, listings_df: pd.DataFrame = None, category=None):
        if listings_df is None:
            listings_df = self.origin_df
        pairs = []
        if category:
            listings_df = listings_df[listings_df["category"] == category].reset_index(drop=True)
        else:
            listings_df = listings_df.reset_index(drop=True)
        for i in range(len(listings_df)):
            for j in range(i + 1, len(listings_df)):
                idx_i, idx_j = listings_df.loc[i, "uniq_id"], listings_df.loc[j, "uniq_id"]
                a_i, a_j = listings_df.loc[i, "aspects"], listings_df.loc[j, "aspects"]
                if listings_df.loc[i, "category"] == listings_df.loc[j, "category"]:
                    categ = listings_df.loc[i, "category"]
                    if listings_df.loc[i, "cluster"] == listings_df.loc[j, "cluster"]:
                        label = 1
                    else:
                        label = 0
                    if i < j:
                        pairs.append((idx_i, idx_j, a_i, a_j, categ, label))
                    else:
                        pairs.append((idx_j, idx_i, a_i, a_j, categ, label))
        return pairs

    def postive_sampling(self, max_size = None):
        if not max_size:
            max_size = int(len(self.origin_df) * 1.5)
        pos_pairs = []
        logging.info("positive sampling...")
        for cluster in self.origin_df["cluster"].value_counts().index:
            pos_pairs.extend(self.make_pairs(listings_df=self.origin_df[self.origin_df["cluster"]==cluster]))
        pos_pairs = set(pos_pairs)
        if len(pos_pairs) > max_size:
            from random import sample
            pos_pairs = sample(pos_pairs, max_size)
        self.total_pos = len(pos_pairs)
        print("pos_pairs", len(pos_pairs))
        return set(pos_pairs)

    def negative_sampling(self):
        neg_pairs = []
        neg_len = self.total_pos * self.negative_ratio
        from src.preprocess.lsh_retriever import LshRetriever
        from random import sample
        lsh_retriever = LshRetriever(in_df=self.origin_df, category=self.category, threshold=0.6, num_perm=16)
        lsh_retriever.bulid_lsh(minhash_col="aspects", separator='|', index_col="uniq_id")

        logging.info("negative sampling...")
        for cluster in self.origin_df["cluster"].value_counts().index:
            key = self.origin_df[self.origin_df["cluster"]==cluster].sample(n=1, axis=0).reset_index(drop=True)
            candidate_idx = lsh_retriever.query(key.loc[0, "uniq_id"])
            candidates = self.make_pairs(listings_df=self.origin_df[self.origin_df["uniq_id"].isin(candidate_idx)])
            candidates = [i for i in candidates if i[-1] == 0]
            candidates = sample(candidates, int(len(candidates)*0.9))
            neg_pairs.extend(candidates)
            if len(set(neg_pairs)) >= neg_len:
                break
        if len(set(neg_pairs)) > neg_len*1.1:
            logging.info(f"assert size of neg_pairs-- actual size: {len(set(neg_pairs))}, expect size: {neg_len}")
            neg_pairs = sample(neg_pairs, neg_len)
        elif len(set(neg_pairs)) < neg_len*0.9:
            logging.info(f"assert size of neg_pairs-- actual size: {len(set(neg_pairs))}, expect size: {neg_len}")
            while len(set(neg_pairs)) < neg_len*0.9:
                candidates = self.origin_df.sample(n=2, axis=0).reset_index(drop=True)
                pair = self.make_pairs(listings_df=candidates)
                if pair[0][-1] == 0:
                    neg_pairs.extend(pair)
        neg_pairs = set(neg_pairs)
        print("neg_pairs", len(neg_pairs))
        return neg_pairs

    def generate_pairset(self, postive_restrict=None, output_file=None):
        pos = self.postive_sampling(max_size=postive_restrict)
        neg = self.negative_sampling()
        pair_set = pd.DataFrame(pos.union(neg))
        pair_set.columns = ["uniq_id_1", "uniq_id_2", "aspects_1", "aspects_2", "category", "label"]
        pair_set.sort_values(by="category", inplace=True)

        if output_file:
            pair_set.to_csv(output_file, sep='\t', index=False)
        return pair_set


if __name__ == '__main__':
    config = Configuration('../../', suffix='flipkart')
    # category: "clothing", "jewellery", "footwear", "mobiles & accessories"

    df = pd.read_csv(config.cleaned_data_file, sep='\t')
    pair_generator = PairsetGenerator(df, category=["clothing", "jewellery", "footwear", "mobiles & accessories"], negative_ratio=5)
    pairset = pair_generator.generate_pairset(postive_restrict=None, output_file=config.pairset_data_file)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.max_colwidth", 200)
    print(pairset.head())
    print("total size", len(pairset))
    print("True size", len(pairset[pairset["label"]==1]))
    print("False size", len(pairset[pairset["label"]==0]))



