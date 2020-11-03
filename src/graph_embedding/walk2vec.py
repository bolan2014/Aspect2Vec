"""
@Time   : 2020-01-13 14:03
@Author : Kyrie.Z
@File   : fasttext_word2vec.py
"""


import fasttext


def walk_to_vec(cf, dim=100, model="skipgram"):
    model = fasttext.train_unsupervised(
        cf.sequences_file,
        model=model,
        dim=dim,
        epoch=50,
        minCount=1
    )

    words = model.get_words()
    hdlr = open(cf.aspect_vec_file, "w", encoding='utf-8')
    hdlr.write("{} {}\n".format(len(words), model.get_dimension()))
    for w in words:
        v = model.get_word_vector(w)
        vstr = w
        for vi in v:
            vstr += " " + str(vi)
        hdlr.write("{}\n".format(vstr))


if __name__ == '__main__':
    pass
