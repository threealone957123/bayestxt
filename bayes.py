def loadDataSet():
    """
    测试数据，每行可看成一个文章包含的词
    返回文章列表及其分类向量
    """
    postingList = [['my','dog','has','flea',\
                    'problems','help','please'],
                   ['maybe','not','take','him',\
                    'to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute',\
                    'I','love','him'], 
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how',\
                    'to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']]
    classVec = [0,1,0,1,0,1]   #1代表侮辱性文字，0代表正常言论
    return postingList,classVec

def creatVocabList(dataSet):
    vocabSet = set([])       #创建一个空集
    for document in dataSet:
        vocabSet = vocabSet | set(document)    #创建两个集合的并集
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)  #创建一个其中所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print ("the word: %s is not in my Vocabulary!" % word)
    return returnVec
    
import bayes 
listOPosts,listClasses = bayes.loadDataSet()
myVocablist = bayes.creatVocabList(listOPosts)
print(myVocablist)
print(bayes.setOfWords2Vec(myVocablist,listOPosts[0]))
print(bayes.setOfWords2Vec(myVocablist,listOPosts[3]))


def trainNB0(trainMatrix,trainCategory):
    """
    :param trainMatrix:
    [
        [1,0,...],
        [0,1,...]
    ]
    :param trainCategory:
    [1,0,...]
    :return:
    """
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    # 不和谐文档的概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # 每个词出现的次数初始化为1，防止后面计算p(x1|c)p(x2|c)...p(xn|c)的时候为0
    p0Num = ones(numWords); p1Num = ones(numWords)
    p0Denom = 2.0; p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            # 向量和，统计类比1下每个词的总数
            p1Num += trainMatrix[i]
            # 得到类别1下的总次数
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    # 避免p(x1|c)p(x2|c)...p(xn|c)得到很小的数最后四舍五入为0
    p1Vect = log(p1Num/p1Denom)
    p0Vect = log(p0Num/p0Denom)
    # 返回p(w|0) p(w|1) p(1) 这里w是向量
    return p0Vect,p1Vect,pAbusive
    
    
from numpy import *
import bayes
import imp
imp.reload(bayes)
listOPosts,listClasses = bayes.loadDataSet()
myVocablist = bayes.creatVocabList(listOPosts)
trainMat=[]
for postinDoc in listOPosts:
    trainMat.append(bayes.setOfWords2Vec(myVocablist,postinDoc))
    
p0V,p1V,pAb=bayes.trainNB0(trainMat,listClasses)
print(pAb)
print(p0V)
print(p1V)