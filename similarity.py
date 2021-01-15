import distance


s1 = '你在干什么'
s2 = '你在干嘛呢'
def jacc_distance(s1,s2):
    first = set(s1).intersection(set(s2))
    second = set(first).union(set(s2))
    return len(first)/len(second)

x = set(list(s1))
y = set(list(s2))

#print(jacc_distance(x,y))
#杰卡德系数计算
#用于比较样本之间的相似性与差异性，系数越大相似度越高
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

def add_space(s):
    return ' '.join(list(s))
def jacc_similarity(s1,s2):
    s1, s2 = add_space(s1), add_space(s2)#将字中间加入空格
    #转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda  s:s.split())
    corpus = [s1,s2]
    vectors = cv.fit_transform(corpus).toarray()
    #求交集
    numerator = np.sum(np.min(vectors, axis = 0))
    #求并集
    denominator = np.sum(np.max(vectors, axis = 0))
    #计算杰卡德系数
    return 1.0*numerator / denominator

#print(jacc_similarity(s1,s2))

#TF 计算TF矩阵中两个向量的相似度，实际上就是求解两个向量夹角的余弦值
    #就是点乘积以二者的模长，公式如下 cos = a*b/|a|*|b|

from scipy.linalg import norm

def tf_similarity(s1,s2):
    #将字中间加入空格
    s1, s2 = add_space(s1), add_space(s2)
    #转化为TF矩阵
    cv = CountVectorizer(tokenizer=lambda  s:s.split())
    corpus = [s1,s2]
    vectors = cv.fit_transform(corpus).toarray()
    #计算TF系数
    return np.dot(vectors[0],vectors[1]) / (norm(vectors[0]) * norm(vectors[1]))

print(tf_similarity(s1,s2))