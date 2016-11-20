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
N=47 #Number of input features



## BATCH WISE MEAN
##def mean_x (x):
##    k=0;
##    z=x[0:int(math.floor((16000/M)*M))];
##    a=np.vsplit(z,M);
##    mu=np.zeros((M,10))
##    sigma=np.zeros((10,10))
##    for i in range(M):
##        for j in range(10):
##            mu[i][j]=np.asarray(a[i]).T[j].mean()
##    return mu

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
    for i in range(55698):
        y_hat=float(np.dot(weight,PHI_train[i]));
        sse+=math.pow((y_hat-float(Y_train[i])),2)
    sse/=2;
    print ("Train ERMS:")
    print (math.sqrt((2*sse)/55698))
    print ("VALIDATION")
    sse=0;
    for i in range(6962):
        y_hat=float(np.dot(weight,PHI_validate[i]));
        sse+=math.pow((y_hat-float(Y_validate[i])),2)
    sse/=2;
    print ("Validate ERMS:")
    print (math.sqrt((2*sse)/6962))
    print ("TEST")
    sse=0;
    count=0;
    for i in range(6962):
        y_hat=float(np.dot(weight,PHI_test[i]));
        sse+=math.pow((y_hat-float(Y_test[i])),2)
    sse/=2;
    print ("Test ERMS:")
    print (math.sqrt((2*sse)/6962))

#############################################



#### MAIN ####


print ("Name: Sugosh Nagavara Ravindra")
print ("Person Number: 50207357")
print ("UBID: sugoshna")

   
X=np.genfromtxt('Querylevelnorm_X.csv',delimiter=',');
Y=np.genfromtxt('Querylevelnorm_t.csv',delimiter=',');    

## BASIS FUNCTION IS ALREADY APPENDED IN X1

## TRAINING = 80%
X_train=X[0:55698]
Y_train=Y[0:55698];

## VALIDATION = 10%
X_validate=X[55698:62660]
Y_validate=Y[55698:62660];

## TESTING = 10%
X_test=X[62660:69623];
Y_test=Y[62660:69623];



#mu=mean_x(X_train);


## K-MEAN CLUSTERING FOR M CLASSES
kmeans = KMeans(n_clusters=M, random_state=0).fit(X_train)
mu = kmeans.cluster_centers_


SIGMA=np.eye(N)
##for i in range(N):
##    SIGMA[i][i]=np.var(mu.T[i])/10.0
##    if SIGMA[i][i]==0:
##        SIGMA[i][i]=0.001
##SIGMA=SIGMA*1000
##SIGMA=np.asmatrix(SIGMA).I

print ("\nM = %d"%M)
print ("Means calculated using K-means clustering")
print ("Lambda = %d"%ld)
print ("Sigma = Identity Matrix")
print ("Learning rate = 0.0001")



## PHI TRAIN-VALIDATION-TEST
PHI_train=phi(X_train,mu,SIGMA);
PHI_validate=phi(X_validate,mu,SIGMA);
PHI_test=phi(X_test,mu,SIGMA);

## WEIGHT CALCULATION USING MOORE PENROSE INVERSE
weight=weight_calc(PHI_train,Y_train);
## CLOSED FORM SOLUTION
closed_form(weight,PHI_train,Y_train,PHI_validate,Y_validate,PHI_test,Y_test)

prev=999.0

## ASSIGN WEIGHT TO ONES FOR GRADIENT DESCENT
weight=np.ones((1,M))
print ("####################################")
print ("\nGRADIENT DESCENT STARTING... %d iterations"%MAX_ITER)
ED=np.zeros(M);
## GRADIENT DESCENT FOR MAX_ITER ITERATIONS
for iterations in range(MAX_ITER):
    print ("\nITERATION %d"%iterations)
    sse=0.0;
    count=0;

    ##UNCOMMENT FOR BATCH GRADIENT DESCENT
    #alpha=-0.1 #LEARNING RATE
    #weight_temp=(alpha*ED)/55698
    #print ""
    #print "WEIGHT_TEMP"
    #print weight_temp
    #weight = weight+weight_temp
    print ("Weight=")
    print (weight)
    print ("TRAINING")
    for i in range(55698):
        y_hat=float(np.dot(weight,PHI_train[i]));
        alpha=1;
        sse+=math.pow((y_hat-float(Y_train[i])),2)
        #if(round(y_hat)==float(Y_train[i])):
        #    count=count+1
        weight_temp=np.zeros(M)
        #ED=np.zeros(M)
        if iterations != MAX_ITER:
            for k in range(M):
                ##UNCOMMENT FOR BATHC GRADIENT DESCENT
                #ED[k]=ED[k]+(float((y_hat)-float(Y_train[i]))*PHI_train[i][k])
                ED[k]=(float((y_hat)-float(Y_train[i]))*PHI_train[i][k])
        #if iterations%10==0:
        weight=weight-0.0001*ED
##        weight=weight-0.006*(ED+ld*weight) #REGULARIZATION
    
    ## ALTERNATE ACCURACY
    ##print "Accuracy="
    ##print float(float(count)/float(55690))*float(100)
        
    sse/=2;
##    print "Train SSE:"
##    print sse
    print ("Train ERMS:")
    print (math.sqrt((2*sse)/55698))
    print ("VALIDATION")
    sse=0;
    count=0;
    for i in range(6962):
        y_hat=float(np.dot(weight,PHI_validate[i]));
        sse+=math.pow((y_hat-float(Y_validate[i])),2)
        #if(round(float(y_hat))==float(Y_validate[i])):
        #    count=count+1
    
    #print "Accuracy="
    #print float(float(count)/float(6962))*float(100)

    sse/=2;
    #print "Validate SSE:"
    #print sse
    print ("Validate ERMS:")
    print (math.sqrt((2*sse)/6962))
    print ("TEST")
    sse=0;
    count=0;
    for i in range(6962):
        y_hat=float(np.dot(weight,PHI_test[i]));
        sse+=math.pow((y_hat-float(Y_test[i])),2)
        #if(round(y_hat)==float(Y_test[i])):
        #    count=count+1
        
    #print "Accuracy=" 
    #print float(float(count)/float(6962))*float(100)
    sse/=2;
    #print "Test SSE:"
    #print sse
    print ("Test ERMS:")
    print (math.sqrt((2*sse)/6962))
    
    ##TO BREAK IF PREVIOUS ERMS IS LESSER THAN CURRENT
    erms=math.sqrt((2*sse)/6962)
    if prev<erms:
        break
    prev=erms
print("--- %s seconds ---" % (time.time() - start_time))

