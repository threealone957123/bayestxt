from random import randrange
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
def generateData():
 '''生成测试数据'''
    def get(start,end)
        return [randrange(start,end) for _ in range(30)]
    
    x1=get(0,40)
    x2=get(70,100)
    y1=get(0,30)
    y2=get(40,70)

    data=list(zip(x1,y1))+\
        list(zip(x1,y2))+\
        list(zip(x2,y1))+\
        list(zip(x2,y2))
    return np.array(data)

    
def AgglomerativeTest(n_clusters):
'''聚类，制定类的数量，并绘制图形'''
    assert 1<=n_clusters<=4
    predictResult=AgglomerativeClustering(n_clusters=n_clusters,affinity='euclidean',linkage='ward').fit_predict(data)
    colors='rgby'
    markers='o*v+'
    for i in range(n_clusters):
        subData=data[predictResult==i]
        plt.scatter(subData[:,0],subData[:,1],c=colors[1],marker=markers[i],s=40)
    plt.show()

    
#生成随机数据
data=generateData()
#聚类为3个不同的类
AgglomerativeClustering(3)
#聚类为4个不同的类
AgglomerativeClustering(4)
