# -*- coding: utf-8 -*-
"""
Created on Sat Jun 30 00:10:01 2018

@author: prometheus
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# extract training, testing data from csv file
def data_extraction():
    train = np.loadtxt(open("train.csv","rb"),delimiter=",",skiprows=0) 
    train_label_true=train[:,0]
    train_data=train[:,1:]
    
    test_data=np.loadtxt(open("test.csv","rb"),delimiter=",",skiprows=0) 
    test_label_true=np.loadtxt(open("label.csv","rb"),delimiter=",",skiprows=0) 
    return train_data,train_label_true,test_data,test_label_true

## inilization done
# LDA dimension reduction 
def LDA_reduction(X,y,X1):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()
    lda.fit(X,y)
    X_new = lda.transform(X)
    X_new1 = lda.transform(X1)
    #lda.transform(X1)
    
    return X_new,X_new1

# PCA dimension reduction
def PCA_reduction(train_data,test_data):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=0.9)
    pca.fit(train_data)
    train_new=pca.transform(train_data)
    test_new=pca.transform(test_data)
    return train_new,test_new

# Tree-based feature selection
def feature_selection(X,y):
    from sklearn.ensemble import ExtraTreesClassifier 
    from sklearn.feature_selection import SelectFromModel
    clf = ExtraTreesClassifier()
    clf = clf.fit(X, y)
    importances=clf.feature_importances_
    indices=np.argsort(importances)[::-1]
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    index=indices[0:len(X_new[0])]
    return X_new,index

# Generate noise from p-norm ball
def noise_pnorm(R,p,n,train_data):
    train_n=np.zeros((200,204))
    for i in enumerate(train_data):
        epsilon=signal.general_gaussian(n, p, sig=p)
        epsilon[abs(epsilon)<1e-5]=0
        sign=[(-1)**np.random.randint(2) for i in range(n)]
        x=np.multiply(epsilon,sign)
        z=(np.random.uniform(0,1))**(1/n)
        x_pnorm=np.linalg.norm(x,ord=p)
        y=R*z*x/x_pnorm
        train_n[i[0]]=train_data[i[0]] + y
    return train_n

#noise optimization
def costfunc(R,p,n,train_data,train_label_true,test_data,test_label_true):
    train_data=noise_pnorm(R,p,n,train_data)
    er_test,er_train=random_forest_classifier(train_data,train_label_true,test_data,test_label_true)
    return er_test
def randomsearch(Rmax,pmax,iter,n,train_data,train_label_true,test_data,test_label_true):
    from sklearn.ensemble import RandomForestClassifier
    p=pmax*np.random.rand(iter)
    R=Rmax*np.random.rand(iter)
    best=1
    for i in range(iter):
        for j in range(iter):
            
            train_gen=noise_pnorm(R[j],p[i],n,train_data)
            clf = RandomForestClassifier(n_estimators=8)
            clf.fit(train_gen,train_label_true)
            test_label_predict=clf.predict(test_data)
            error_test=test_label_predict-test_label_true!=0
            er_test=np.sum(error_test)/len(test_label_true)
            cost=er_test
#            print(train_gen[95:105,100])
            if cost<best:
                best=cost
                bestp=p[i]
                bestR=R[j]
                train_out=train_gen
        print (i,best)
    
    return bestp,bestR,best,train_out

def cost_func(z):
    R,p=z
    train_gen=noise_pnorm(R,p,n,train_data)
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=8)
    clf.fit(train_gen,train_label_true)
    test_label_predict=clf.predict(test_data)
    error_test=test_label_predict-test_label_true!=0
    er_test=np.sum(error_test)/len(test_label_true)
    return er_test

# compute the train and test error
def compute_error(clf):
    # training error
    train_label_predict=clf.predict(train_data)
    error_train=train_label_predict-train_label_true!=0
    er_train=np.sum(error_train)/len(train_label_true)
    # test error
    test_label_predict=clf.predict(test_data)
    error_test=test_label_predict-test_label_true!=0
    er_test=np.sum(error_test)/len(test_label_true)
    return er_test,er_train  

# Multinomial Naive Bayes Classifier
def naive_bayes_classifier(train_data,train_label_true,test_data,test_label_true):
    from sklearn.naive_bayes import MultinomialNB
    clf = MultinomialNB(alpha=0.01)
    clf.fit(train_data, train_label_true)  
    er_test,er_train=compute_error(clf)
    return er_test,er_train

# KNN Classifier
def knn_classifier(train_data,train_label_true,test_data,test_label_true):
    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier()
    clf.fit(train_data, train_label_true)  
    er_test,er_train=compute_error(clf)
    return er_test,er_train

# Logistic regression classifier
def logistic_regression_classifier(train_data,train_label_true,test_data,test_label_true):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty='l2')
    clf.fit(train_data, train_label_true)  
    er_test,er_train=compute_error(clf)
    return er_test,er_train

# Random Forest Classifier
def random_forest_classifier(train_data,train_label_true,test_data,test_label_true):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=8)
    clf.fit(train_data,train_label_true)
    er_test,er_train=compute_error(clf)
    return er_test,er_train       

# Decision Tree Classifier
def decision_tree_classifier(train_data,train_label_true,test_data,test_label_true):
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf.fit(train_data,train_label_true)
    er_test,er_train=compute_error(clf)
    return er_test,er_train 

# GBDT(Gradient Boosting Decision Tree) Classifier
def gradient_boosting_classifier(train_data,train_label_true,test_data,test_label_true):
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=200)
    clf.fit(train_data,train_label_true)
    er_test,er_train=compute_error(clf)
    return er_test,er_train 

# Support Vector Machine classifier     
def svm_classifier(train_data,train_label_true,test_data,test_label_true):
    from sklearn import svm
    clf = svm.SVC()
    clf.fit(train_data, train_label_true)  
    er_test,er_train=compute_error(clf)
    return er_test,er_train

# SVM Classifier using cross validation
def svm_cross_validation(train_data,train_label_true,test_data,test_label_true):
    from sklearn.grid_search import GridSearchCV
    from sklearn.svm import SVC
    clf = SVC(kernel='rbf', probability=True)
    param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000], 'gamma': [0.001, 0.0001]}
    grid_search = GridSearchCV(clf, param_grid, n_jobs = 1, verbose=1)
    grid_search.fit(train_data, train_label_true)
    best_parameters = grid_search.best_estimator_.get_params()
    for para, val in best_parameters.items():
        print (para, val)
    clf = SVC(kernel='rbf', C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
    clf.fit(train_data, train_label_true)
    er_test,er_train=compute_error(clf)
    return er_test,er_train

# coding: utf-8

import random   

 
#----------------------PSO参数设置---------------------------------  
class PSO():  
    def __init__(self,pN,dim,max_iter):  
        self.w = 0.8    
        self.c1 = 2     
        self.c2 = 2     
        self.r1= 0.6  
        self.r2=0.3  
        self.pN = pN                #粒子数量  
        self.dim = dim              #搜索维度  
        self.max_iter = max_iter    #迭代次数  
        self.X = np.zeros((self.pN,self.dim))       #所有粒子的位置和速度  
        self.V = np.zeros((self.pN,self.dim))  
        self.pbest = np.zeros((self.pN,self.dim))   #个体经历的最佳位置和全局最佳位置  
        self.gbest = np.zeros((1,self.dim))  
        self.p_fit = np.zeros(self.pN)              #每个个体的历史最佳适应值  
        self.fit = 1e10             #全局最佳适应值  
          
#---------------------目标函数Sphere函数-----------------------------  
    def function(self,z):  
        R,p=z
        train_gen=noise_pnorm(R,p,n,train_data)
        from sklearn import tree
        clf = tree.DecisionTreeClassifier()
#        from sklearn import svm
#        clf = svm.SVC()
        clf.fit(train_gen,train_label_true)
        test_label_predict=clf.predict(test_data)
        error_test=test_label_predict-test_label_true!=0
        er_test=np.sum(error_test)/len(test_label_true)
        return er_test,train_gen
        
        
        
#        sum = 0  
#        length = len(x)  
#        x = x**2  
#        for i in range(length):  
#            sum += x[i]  
#        return sum
#---------------------初始化种群----------------------------------  
    def init_Population(self):  
        for i in range(self.pN):  
            for j in range(self.dim):  
                self.X[i][j] = random.uniform(0,20)  
                self.V[i][j] = random.uniform(0,5)  
            self.pbest[i] = self.X[i]  
            tmp,train_gen = self.function(self.X[i])  
            self.p_fit[i] = tmp  
            if(tmp < self.fit):  
                self.fit = tmp  
                self.gbest = self.X[i]  
      
#----------------------更新粒子位置----------------------------------  
    def iterator(self):  
        fitness = []  
        Gbest=[]
        for t in range(self.max_iter):  
            for i in range(self.pN):         #更新gbest\pbest  
               temp,train_gen = self.function(self.X[i])  
               if(temp<self.p_fit[i]):      #更新个体最优  
                   self.p_fit[i] = temp  
                   self.pbest[i] = self.X[i]  
                   if(self.p_fit[i] < self.fit):  #更新全局最优  
                       self.gbest = self.X[i]  
                       self.fit = self.p_fit[i] 
                       train_out=train_gen
            for i in range(self.pN):  
                self.V[i] = self.w*self.V[i] + self.c1*self.r1*(self.pbest[i] - self.X[i]) + \
                            self.c2*self.r2*(self.gbest - self.X[i])  
                self.X[i] = self.X[i] + self.V[i]  
                
#                print (self.X[i])
                if self.X[i][0]<=0:
                    self.X[i][0]=random.uniform(0,10)
                if self.X[i][1]<=0:
                    self.X[i][1]=random.uniform(0,10) 
            fitness.append(self.fit)  
            Gbest.append(self.gbest)
            print(t,self.fit)                   #输出最优值  
        return fitness,Gbest,train_out
 



if __name__ == '__main__':

    ## data extraction from csv file
    train_data,train_label_true,test_data,test_label_true=data_extraction()
    true=test_label_true

    ## data augmentation-- adding uniform noise from a p-norm ball

    
#    train_data=train_data+np.random.uniform(-R,R,(len(train_data), len(train_data[0])))
#    R=float(10);
#    p=float(10);
    n=len(train_data[0])
#    train_data=noise_pnorm(R,p,n,train_data,train_label_true,test_data,test_label_true)
    
    ## Feature selection or demionsion reduciton
#    tree based feature selection
#    
#    train_data,index=feature_selection(train_data,train_label_true)  
#    test_data=test_data[:,index]
#    
#    PCA
    
#    train_data,test_data=PCA_reduction(train_data,test_data)
#    
#    LDA
#    train_data,test_data=LDA_reduction(train_data,train_label_true,test_data)

#    Rmax=10
#    pmax=10
#    iter=10
#    bestp,bestR,best,train_data=randomsearch(Rmax,pmax,iter,n,train_data,train_label_true,test_data,test_label_true)
#    backup_t=train_data
    
    
    #----------------------PSO-----------------------  
    my_pso = PSO(pN=30,dim=2,max_iter=100)  
    my_pso.init_Population()  
    fitness,Gbest,train_out = my_pso.iterator()    
    train_data=train_out
    ## starting classifying 
    max_iter=25
    current_iter=1 
    Error_test=[]
    Index=[]
    Train=[]
    Test=[]
    while current_iter<max_iter:
        current_iter += 1   
#        from sklearn.ensemble import RandomForestClassifier
#        clf = RandomForestClassifier(n_estimators=8)
        from sklearn import tree
        clf = tree.DecisionTreeClassifier()
        clf.fit(train_data,train_label_true) 
        er_test,er_train=compute_error(clf)
        train_label_predict=clf.predict(train_data)
        error_train=train_label_predict-train_label_true!=0
        er_train=np.sum(error_train)/len(train_label_true)
        # test error
        test_label_predict=clf.predict(test_data)
        error_test=test_label_predict-test_label_true!=0
        er_test=np.sum(error_test)/len(test_label_true)
        print('Test error is:')
        print(er_test)
        
        index=np.where(error_test)[0]
        index_add=np.random.choice(index, 400,replace=False)
#        index_add=np.sort(index_add)
#        index_add=index[0:399]
        train_data=np.concatenate([train_data,test_data[index_add]])
        train_label_true=np.concatenate([train_label_true,test_label_predict[index_add]])
        test_data=np.delete(test_data,index_add,axis=0)    
        test_label_true=np.delete(test_label_true,index_add,axis=0) 
        Error_test.append(er_test)
        Index.append(index_add)
        Train.append(train_label_predict)
        Test.append(test_label_predict)
        


    x=list(range(0, 24*400,400))    
    plt.plot(x,Error_test)
    plt.xlabel("Number of unlabeled samples", size=14)  
    plt.ylabel("Error", size=14)  

    plt.ylim([0,1])
    plt.xlim([0,10000])
    
## 还原    
#    for i in range(0,23):
#        label_400=Train[23-i][-400:]
#        ind=Index[22-i]
#        current=np.zeros((len(Test[22-i]),))
#        current[ind]=label_400
#        k=0
#        for j in range(0,len(Test[23-i])):
#            if j in Index[22-i]:
#                k += 1
#            else:
#                current[k]=Test[23-i][j]
#                k += 1        
#    eroutput= current-true != 0
#    output=np.sum(eroutput)/20000
    
    
#    test_classifiers = ['KNN', 'LR', 'RF', 'DT', 'SVM', 'GBDT']
#    classifiers = {'NB':naive_bayes_classifier, 
#                  'KNN':knn_classifier,
#                   'LR':logistic_regression_classifier,
#                   'RF':random_forest_classifier,
#                   'DT':decision_tree_classifier,
#                  'SVM':svm_classifier,
#                'SVMCV':svm_cross_validation,
#                 'GBDT':gradient_boosting_classifier
#    }  
#
#    for classifier in test_classifiers:
#        # test and train error stored in a tuple respectively 
#        locals()['er_'+classifier]=classifiers[classifier](train_data,train_label_true,test_data,test_label_true)


    