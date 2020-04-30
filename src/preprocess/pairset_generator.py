import pandas as pd
import logging
from tqdm import tqdm
from src.utils.config import Configuration

class PairsetGenerator(object):
    def __init__(self, df, category: list, negative_ratio=5):
        self.origin_df = df[df["category"].isin(category)].reset_index()
        self.negative_ratio=negative_ratio
        self.category = category
        self.total_pos = 1000

    def make_pairs(self, listings_df: pd.DataFrame = None, category=None):
        if listings_df is None:
            listings_df = self.origin_df
        pairs = []
        if category:
            listings_df = listings_df[listings_df["category"] == category].reset_index()
        else:
            listings_df = listings_df.reset_index()
        for i in range(len(listings_df)):
            for j in range(i + 1, len(listings_df)):
                a_i, a_j = listings_df.loc[i, "aspects"], listings_df.loc[j, "aspects"]
                if listings_df.loc[i, "cluster"] == listings_df.loc[j, "cluster"]:
                    label = 1
                else:
                    label = 0
                if i < j:
                    pairs.append((a_i, a_j, label))
                else:
                    pairs.append((a_j, a_i, label))
        return pairs

    def postive_sampling(self):
        pos_pairs = []
        logging.info("positive sampling...")
        for cluster in self.origin_df["cluster"].value_counts().index:
            pos_pairs.extend(self.make_pairs(listings_df=self.origin_df[self.origin_df["cluster"]==cluster]))
        pos_pairs = set(pos_pairs)
        self.total_pos = len(pos_pairs)
        return pos_pairs

    def negative_sampling(self):
        neg_pairs = []
        neg_len = self.total_pos * self.negative_ratio

        from src.preprocess.lsh_retriever import LshRetriever
        lsh_retriever = LshRetriever(in_df=df, category=self.category, threshold=0.6, num_perm=16)
        lsh_retriever.bulid_lsh(minhash_col="aspects", separator='|', index_col="uniq_id")

        logging.info("negative sampling...")
        for cluster in self.origin_df["cluster"].value_counts().index:
            key = self.origin_df[self.origin_df["cluster"]==cluster].sample(n=1, axis=0).reset_index()
            candidate_idx = lsh_retriever.query(key.loc[0, "uniq_id"])
            candidates = [i for i in self.make_pairs(listings_df=self.origin_df[self.origin_df["uniq_id"].isin(candidate_idx)]) if i[2]==0]
            neg_pairs.extend(candidates)
            if len(set(neg_pairs)) >= neg_len:
                break
        neg_pairs = set(neg_pairs)
        return neg_pairs

    def generate_pairset(self):
        pos = self.postive_sampling()
        neg = self.negative_sampling()
        pair_set = pos.union(neg)
        return list(pair_set)




if __name__ == '__main__':
    config = Configuration('../../', suffix='flipkart')
    df = pd.read_csv(config.cleaned_data_file, sep='\t')
    pair_generator = PairsetGenerator(df, category=["clothing", "footwear", "jewellery"], negative_ratio=5)
    pairset = pair_generator.generate_pairset()
    print("total len", len(pairset))
    print("false len", len([i for i in pairset if i[2]==1]))






