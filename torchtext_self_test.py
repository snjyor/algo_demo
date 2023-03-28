import re
from sklearn.decomposition import PCA
import jieba
from gensim.models import Word2Vec
from gensim.models import FastText
import matplotlib.pyplot as plt
import matplotlib
from sklearn.manifold import TSNE


def get_data():
    # 得到二维数据
    with open("data/sanguo.txt", "r") as file:
        data = file.readlines()
    all_words = []
    for row in data:
        res = jieba.cut(row.replace("\n", ""))
        words = []
        for each_word in res:
            use_word = re.sub("[\s+\.\!\/_,$%^*(+\"\'””《》]+|[+——！，。？、~@#￥%……&*（）：；‘]+-", "", each_word)
            if use_word:
                words.append(use_word)
        if words:
            all_words.append(words)
    return all_words


def text_train(all_words):
    # 训练得到向量模型
    model = Word2Vec(sentences=all_words, window=8, min_count=3, negative=5, sg=0, epochs=10, vector_size=20)
    # model = FastText(sentences=all_words, window=8, min_count=3, negative=5, sg=0, epochs=10, vector_size=20)
    km_array = model.wv.get_vector('孔明')
    print(f"孔明的向量表达：\n{km_array}")
    km_most_simi = model.wv.most_similar(positive=["刘备"])
    print(km_most_simi)
    return model


def simi_text(model):
    # 类比验证
    words = model.wv.most_similar(positive=['刘备'])
    print(words)


def vision_text(model):
    # 降维可视化
    raw_list = model.wv.vectors
    # two_dim_array = PCA(n_components=2).fit_transform(raw_list)
    two_dim_array = TSNE(n_components=2).fit_transform(raw_list)
    print(two_dim_array)
    plt.figure(figsize=(20, 8), dpi=100)
    plt.plot(two_dim_array[:, 0], two_dim_array[:, 1], ".")
    text_dict = model.wv.key_to_index
    words = ['孙权', '刘备', '曹操', '周瑜', '诸葛亮', '司马懿', '汉献帝']
    zhfont1 = matplotlib.font_manager.FontProperties(fname='./华文仿宋.ttf', size=16)
    for word in words:
        for key, value in text_dict.items():
            if word == key:
                xy = two_dim_array[value]
                plt.plot(xy[0], xy[1], '.', alpha=1, color='orange', markersize=10)
                plt.text(xy[0], xy[1], word, fontproperties=zhfont1)
    plt.show()


if __name__ == '__main__':
    all_words = get_data()
    model = text_train(all_words)
    simi_text(model)
    vision_text(model)



