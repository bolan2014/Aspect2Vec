from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
import pickle
import logging
import os


class LshRetriever(object):
    def __init__(self, in_df, category: list, threshold=0.6, num_perm=16):
        self.category = category
        self.num_perm = num_perm
        self.threshold = threshold
        self.in_df = in_df[in_df["category"].isin(self.category)].reset_index()
        self.lsh = MinHashLSH(threshold=self.threshold, num_perm=self.num_perm)

    @staticmethod
    def generate_minhash(in_str, separator='|', num_perm=16, return_hashvalue=False):
        m = MinHash(num_perm=num_perm)
        in_set = in_str.split(separator)
        # for aspt in in_set:
        #     _in_set.extend(aspt.split())
        for s in set(in_set):
            m.update(s.encode('utf8'))
        if return_hashvalue:
            return ','.join(map(str, m.hashvalues))
        else:
            return m

    def bulid_lsh(self, minhash_col="aspects", separator='|', index_col="uniq_id"):
        tqdm.pandas()
        self.index_col = index_col
        self.separator = separator
        self.minhash_col = minhash_col
        logging.info("building lsh...")
        self.in_df["minhash"] = self.in_df.progress_apply(
                                lambda row: self.generate_minhash(row[minhash_col], separator=separator, num_perm=self.num_perm),
                                axis=1)
        data_list = zip(self.in_df[index_col].astype(str), self.in_df["minhash"])
        for key, minhash in tqdm(data_list):
            self.lsh.insert(key, minhash)

    def query(self,  m_index):
        self.in_df[self.index_col] = self.in_df[self.index_col].astype(str)
        m_query = self.in_df[self.in_df[self.index_col] == str(m_index)].reset_index(drop=True).loc[0, "minhash"]
        if m_index in self.lsh:
            result = self.lsh.query(m_query)
            return result
        else:
            logging.error("query not in lsh")
            return None

    def dump_lsh(self, file_name):
        with open(os.path.join("../../cache", file_name+('_'.join(self.category))+'_'+str(self.threshold)+".pkl"), 'wb') as fw:
            pickle.dump(self.lsh, fw)

    def load_lsh(self, file_name):
        with open(os.path.join("../../cache", file_name+('_'.join(self.category))+'_'+str(self.threshold)+".pkl"), 'rb') as fr:
            self.lsh = pickle.load(fr)