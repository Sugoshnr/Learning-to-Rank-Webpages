import numpy as np
import math
import time
import random
from sklearn.cluster import KMeans



start_time = time.time()

global M,ld,alpha,MAX_ITER,N
M=10 #Number of basis function
ld=0 #Lambda
MAX_ITER=100 #Maximum number of iterations
N=10 #Number of input features



############################################
#PHI CALCULATION
def phi(X_train,mu,SIGMA):
    PHI=np.zeros((len(X_train),len(mu)));
    for i in range(len(X_train)):
        for j in range(len(mu)):
            PHI[i][j]=phi_calc(X_train[i],mu[j],SIGMA)
    return PHI;



def phi_calc(x_1,mu,SIGMA):
    a=np.subtract(x_1,mu);
    #temp=np.dot(a.T,SIGMA) ##Uncomment for Covariance matrix
    #a=np.dot(temp,a.T)     ##
    #w/o SIGMA              
    a=np.dot(a.T,a)         ##Comment for Covariance matrix
    a=a*(-0.5)
    return math.exp(a)

##############################################
##WEIGHT CALCULATION
def weight_calc(PHI,Y):
    #lamda=ld*np.identity(M)
    a=np.dot(PHI.T,PHI);
    #a=np.add(lamda,a)
    a_mat=np.asmatrix(a);
    a_mat=a_mat.I;
    rc=np.dot(PHI.T,Y)
    weight=np.dot(a_mat,rc);
    return weight;

##############################################


##CLOSED FORM SOLUTION##
def closed_form(weight,PHI_train,Y_train,PHI_validate,Y_validate,PHI_test,Y_test):
    print ("####################################")
    print ("CLOSED FORM SOLUTION")
    sse=0;
    for i in range(16000):
        y_hat=float(np.dot(weight,PHI_train[i]));
        sse+=math.pow((y_hat-float(Y_train[i])),2)
    sse/=2;
    print ("Train ERMS:")
    print (math.sqrt((2*sse)/16000))
    print ("VALIDATION")
    sse=0;
    for i in range(2000):
        y_hat=float(np.dot(weight,PHI_validate[i]));
        sse+=math.pow((y_hat-float(Y_validate[i])),2)
    sse/=2;
    print ("Validate ERMS:")
    print (math.sqrt((2*sse)/2000))
    print ("TEST")
    sse=0;
    count=0;
    for i in range(2000):
        y_hat=float(np.dot(weight,PHI_test[i]));
        sse+=math.pow((y_hat-float(Y_test[i])),2)
    sse/=2;
    print ("Test ERMS:")
    print (math.sqrt((2*sse)/2000))

#############################################



## MAIN ##
    
X=np.genfromtxt('input.csv',delimiter=',');
Y=np.genfromtxt('output.csv',delimiter=',');    

#X=np.concatenate((np.ones((len(X),1)), X),axis=1)

print ("Name: Sugosh Nagavara Ravindra")
print ("Person Number: 50207357")
print ("UBID: sugoshna")


Y=np.asarray(Y)
X=np.asarray(X)
print (Y.shape)
X_train=X[0:16000]
Y_train=Y[0:16000];


X_validate=X[16000:18000]
Y_validate=Y[16000:18000];


X_test=X[18000:20000];
Y_test=Y[18000:20000];


#mu,sigma=mean_x(X_train);

#sigmaInv=np.asmatrix(sigma).I

kmeans = KMeans(n_clusters=M, random_state=0).fit(X_train)
mu = kmeans.cluster_centers_

SIGMA=np.zeros((N,N))
for i in range(N):
    SIGMA[i][i]=np.var(mu.T[i])/10.0
    if SIGMA[i][i]==0:
        SIGMA[i][i]=0.001
SIGMA=SIGMA*1000
SIGMA=np.asmatrix(SIGMA).I
PHI_train=phi(X_train,mu,SIGMA);
PHI_validate=phi(X_validate,mu,SIGMA);
PHI_test=phi(X_test,mu,SIGMA);
weight=weight_calc(PHI_train,Y_train);

print ("\nM = %d"%M)
print ("Means calculated using K-means clustering")
print ("Lambda = %d"%ld)
print ("Sigma = Identity Matrix")
print ("Learning rate = 0.001")


closed_form(weight,PHI_train,Y_train,PHI_validate,Y_validate,PHI_test,Y_test)
print ("Weight = ")
print (weight)

weight=np.ones((1,M))
print ("####################################")
print ("\nGRADIENT DESCENT STARTING... %d iterations"%MAX_ITER)
ED=np.zeros(M);
prev=999.0
for iterations in range(MAX_ITER):
    print ("\nITERATION %d"%iterations)
    sse=0.0;
    count=0;
    #print ED
    #weight_temp=(alpha*ED)/55690
    #print ""
    #print "WEIGHT_TEMP"
    #print weight_temp
    #weight = weight+weight_temp
    print ("Weight:")
    print (weight)
    for i in range(16000):
        y_hat=float(np.dot(weight,PHI_train[i]));
        alpha=1;
        sse+=math.pow((y_hat-float(Y_train[i])),2)
        if(round(y_hat)==float(Y_train[i])):
            count=count+1
        weight_temp=np.zeros(M)
        #ED=np.zeros(M)
        if iterations != MAX_ITER:
            for k in range(M):
                #ED[k]=ED[k]+(float((y_hat)-float(Y_train[i]))*PHI_train[i][k])
                ED[k]=(float((y_hat)-float(Y_train[i]))*PHI_train[i][k])
        #if iterations%10==0:
        weight=weight-0.001*ED
    #print "Accuracy="
    print (float(float(count)/float(16000))*float(100))
        
    sse/=2;
##    print "Train SSE:"
##    print sse
    print ("Train ERMS:")
    print (math.sqrt((2*sse)/16000))
    print ("VALIDATION")
    sse=0;
    count=0;
    for i in range(2000):
        y_hat=float(np.dot(weight,PHI_validate[i]));
        sse+=math.pow((y_hat-float(Y_validate[i])),2)
        if(round(float(y_hat))==float(Y_validate[i])):
            count=count+1
    #print "Accuracy="
    #print float(float(count)/float(6962))*float(100)

    sse/=2;
    #print "Validate SSE:"
    #print sse
    print ("Validate ERMS:")
    print (math.sqrt((2*sse)/2000))
    print ("TEST")
    sse=0;
    count=0;
    for i in range(2000):
        #y_hat,PHI=predict(weight,X_test,i,mu)
        y_hat=float(np.dot(weight,PHI_test[i]));
        sse+=math.pow((y_hat-float(Y_test[i])),2)
        if(round(y_hat)==float(Y_test[i])):
            count=count+1
    #print "Accuracy=" 
    #print float(float(count)/float(6962))*float(100)
    sse/=2;
    #print "Test SSE:"
    #print sse
    print ("Test ERMS:")
    print (math.sqrt((2*sse)/2000))
    erms=math.sqrt((2*sse)/2000)
    if prev<erms:
        break
    prev=erms

print("--- %s seconds ---" % (time.time() - start_time))

















