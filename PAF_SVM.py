# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 14:24:51 2019

@author: think
"""

# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.font_manager
from pyecharts import Scatter3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm


inputfile1 = 'D:/Python/2nd Plant/PAF/PA4B.txt'#2017年全年数据
inputfile2 = 'D:/Python/2nd Plant/PAF/PA4B1213.txt'#2018年12月13日当天数据
#inputfile2 = 'D:/Python/2nd Plant/PAF/PA4B201812.txt'#2018年12月数据
#inputfile2 = 'D:/Python/2nd Plant/PAF/PA4B201811.txt'#2018年11月27-11月30日数据
#inputfile2 = 'D:/Python/2nd Plant/PAF/PA4B201801to08.txt'#2018年1月至8月数据 8月25日后进行B修


data1 = pd.read_csv(inputfile1, sep = ',',header = 0, names = [u'时间',u'负荷',u'煤量',u'电流'])
data2 = pd.read_csv(inputfile2,sep = ',',header = 0,names = [u'时间',u'负荷',u'煤量',u'电流'])
#data3 = pd.read_csv(inputfile3,sep = ',',header = 0,names = [u'时间',u'负荷',u'煤量',u'电流'])
import time

data1['时间'] = data1['时间'].apply(lambda i: time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(i)))
data2['时间'] = data2['时间'].apply(lambda i: time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(i)))
#data3['时间'] = data3['时间'].apply(lambda i: time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(i)))


#data2 = pd.concat([data2,data3],ignore_index = True,axis = 0)

data_train = data1[['负荷','煤量','电流','时间']]
data_test = data2[['负荷','煤量','电流','时间']]

data_train_1 = data_train.query('负荷>200 and 负荷<670')
data_test_1 = data_test.query('负荷>200 and 负荷<670')

index_train = data_train_1.index.values
index_test = data_test_1.index.values

data_mean = (data_train_1.loc[:,['负荷','煤量','电流']].mean()).values
data_std = (data_train_1.loc[:,['负荷','煤量','电流']].std()).values

data_train_1_zs = (data_train_1.values[:,:3]-data_mean)/data_std
data_test_1_zs = (data_test_1.values[:,:3]-data_mean)/data_std


from sklearn.decomposition import PCA

pca1 = PCA(n_components=2)
pca1.fit(data_train_1_zs)
X = pca1.transform(data_train_1_zs)

pca2 = PCA(n_components=2)
pca2.fit(data_test_1_zs)
Y = pca2.transform(data_test_1_zs)

############建模##################

clf = svm.OneClassSVM(nu=0.0000001,kernel = 'rbf',gamma = 0.15)
clf.fit(X)


"""
###########Training Datas Dealing################################
X_pred = clf.predict(X)
X_pca = pd.DataFrame(data = np.c_[X,X_pred],index = index_train)
Time_train = pd.DataFrame(data = data_train_1.loc[index_train,'时间'],index = index_train)
X_pca_WithTime = pd.concat([X_pca,Time_train],axis = 1)

X_abnormal_pca = X_pca_WithTime[X_pca_WithTime.values[:,2]==-1]
X_normal_pca = X_pca_WithTime[X_pca_WithTime.values[:,2]==1]

X_abnormal_index = X_abnormal_pca.index.values
X_normal_index = X_normal_pca.index.values


X_Orignal_abnormal = data_train_1.loc[X_abnormal_index]
X_Orignal_normal = data_train_1.loc[X_normal_index]

"""

###################Test Datas Dealing######################
Y_pred = clf.predict(Y)
Y_pca = pd.DataFrame(data = np.c_[Y,Y_pred],index = index_test)
Time_test = pd.DataFrame(data = data_test_1.loc[index_test,'时间'],index = index_test)
Y_pca_WithTime = pd.concat([Y_pca,Time_test],axis = 1)

Y_abnormal_pca = Y_pca_WithTime[Y_pca_WithTime.values[:,2]==-1]
Y_normal_pca = Y_pca_WithTime[Y_pca_WithTime.values[:,2]==1]

Y_abnormal_index = Y_abnormal_pca.index.values
Y_normal_index = Y_normal_pca.index.values


Y_Orignal_abnormal = data_test_1.loc[Y_abnormal_index]
Y_Orignal_normal = data_test_1.loc[Y_normal_index]




################2D Plot###################

xx,yy = np.meshgrid(np.linspace(-10,10,500), np.linspace(-10,10,500))
Z = clf.decision_function(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)

#plt.title('Novelty Detection')
plt.figure(figsize=(10,8))
plt.contourf(xx,yy,Z,levels = np.linspace(Z.min(),0,5),cmap = plt.cm.PuBu)
a = plt.contour(xx,yy,Z,levels = [0],linewidths = 2,colors = 'darkred')
plt.contourf(xx,yy,Z,levels = [0,Z.max()],colors = 'palevioletred')

s = 40
#b1 = plt.scatter(X_normal_pca.values[:,0],X_normal_pca.values[:,1],c='white',s=s,edgecolors='k')
#b2 = plt.scatter(X_abnormal_pca.values[:,0],X_abnormal_pca.values[:,1],c='gold',s=s,edgecolors='k')
b1 = plt.scatter(Y_normal_pca.values[:,0],Y_normal_pca.values[:,1],c='white',s=s,edgecolors='k')
b2 = plt.scatter(Y_abnormal_pca.values[:,0],Y_abnormal_pca.values[:,1],c='gold',s=s,edgecolors='k')

#c = plt.scatter(Y_outliers[:,0],Y_outliers[:,1],c='gold',s=s,edgecolors = 'k')

plt.axis('tight')
plt.xlim((-10,10))
plt.ylim((-10,10))
plt.legend([a.collections[0],b1,b2],
           ['learned frontier','normal observations',
            'abnormal observations','new abnormal observations'],
           loc = 'upper left',
           prop = matplotlib.font_manager.FontProperties(size=11))
#plt.xlabel(
#        'error train: %d/200; errors novel regular: %d/40;'
#        'errors novel abnormal: %d/40'
#        % (n_error_train,n_error_test,n_error_outliers))

plt.show()


"""
#################3D Plot###################

xx,yy = np.meshgrid(np.linspace(-5,5,500), np.linspace(-5,5,500))
Z = clf.decision_function(np.c_[xx.ravel(),yy.ravel()])
Z = Z.reshape(xx.shape)

fig = plt.figure(figsize = (15,10))
ax = plt.axes(projection = '3d')

Y_test_zs = pd.DataFrame(data = data_test_1_zs,index = index_test)
Y_test_zs_abnormal = Y_test_zs.loc[Y_abnormal_index]
Y_test_zs_normal = Y_test_zs.loc[Y_normal_index]


#X_train_zs = pd.DataFrame(data = data_train_1_zs,index = index_train)
#X_train_zs_abnormal = X_train_zs.loc[X_abnormal_index]
#X_train_zs_normal = X_train_zs.loc[X_normal_index]


contour = ax.contour(xx,yy,Z,extend3d = True, cmap = cm.coolwarm,alpha=0.2)
ax.clabel(contour, fontsize=9, inline=1)
ax.scatter3D(Y_test_zs_normal.values[:,0],Y_test_zs_normal.values[:,1],Y_test_zs_normal.values[:,2],s = 40,c = 'white',edgecolors = 'k')
ax.scatter3D(Y_test_zs_abnormal.values[:,0],Y_test_zs_abnormal.values[:,1],Y_test_zs_abnormal.values[:,2],s = 40,c = 'gold',edgecolors = 'k')

#ax.scatter3D(X_train_zs_normal.values[:,0],X_train_zs_normal.values[:,1],X_train_zs_normal.values[:,2],s = 40,c = 'white',edgecolors = 'k')
#ax.scatter3D(X_train_zs_abnormal.values[:,0],X_train_zs_abnormal.values[:,1],X_train_zs_abnormal.values[:,2],s = 40,c = 'gold',edgecolors = 'k')
#surf = ax.plot_surface(xx,yy,Z,cmap = cm.coolwarm,linewidth = 0,antialiased=False)

# Customize the z axis.
ax.set_zlim(-5,5)
ax.zaxis.set_major_locator(LinearLocator(5))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

# Add a color bar which maps values to colors.
#fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()
"""
