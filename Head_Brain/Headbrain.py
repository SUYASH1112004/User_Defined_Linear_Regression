'''
Classifier : Linear Regression
DataSet    : Head Brain Dataset
Features   : Gender , Age , Head size , Brain Weight
labels     : -
Training Dataset : 237
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def HeadBrainPredictor():
    #Load Data
    data=pd.read_csv('MarvellousHeadBrain.csv')

    print("Size Of Data :",data.shape)

    x= data['Head Size(cm^3)'].values
    y= data['Brain Weight(grams)'].values

    #Least Square Method 
    mean_x=np.mean(x)   #Removing Average 
    mean_y=np.mean(y)   #Removing Average 

    n=len(x)

    numerator=0
    denominator=0
    for i in range(n):
        numerator += (x[i]-mean_x) * (y[i]-mean_y)
        denominator+=(x[i]-mean_x)**2

    m=numerator/denominator

    c=mean_y-(m*mean_x)

    print("Slope Of Regression is :",m)
    print ("Y intercept of regression is ",c)

    max_x=np.max(x)+100
    min_x=np.min(x)-100

    #Displaying Ploting of above points
    X=np.linspace(min_x,max_x,n)

    Y=c+m*X

    plt.plot(X,Y,color='#58b970',label='Regression Line')
    plt.scatter(X,Y,color='#ef5423',label='Scatter plot')
    plt.xlabel('Head Size in cm3')
    plt.ylabel('Brain Weight in gram')

    plt.legend()
    plt.show()

    #FInding out goodness of fit i.e r square method
    ss_t=0
    ss_r=0

    for i in range (n):
        y_pred=c+m*x[i]
        ss_t+=(y[i] - mean_y)**2
        ss_r+=(y_pred-y[i])**2

    r2=1-(ss_r/ss_t)
    print(r2)

def main():
    print("---ML Linear Regression ------")
    print("-------Supervised Machine Learning --------")
    print("Linear Regression on Head and Brain size data set")

    HeadBrainPredictor()

if __name__ =="__main__":
    main()


