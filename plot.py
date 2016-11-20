from matplotlib import pyplot as plt
import numpy as np



f=open("M_ALL_REAL.txt","r")
x=[]
y1=[]
y2=[]
y3=[]
for line in f:
    X=line.split()
    x.append(float(X[0]))
    y1.append(float(X[1]))
    y2.append(float(X[2]))
    y3.append(float(X[3]))

plt.figure(1)
plt.title("M - LeToR Dataset")
plt.xlabel("M")
plt.ylabel("ERMS")
a,=plt.plot(x,y1,label="Train")
b,=plt.plot(x,y2,label="Validation")
c,=plt.plot(x,y3,label="Testing")
plt.legend(handles=[a,b,c])
#for a,b in zip(x, y): 
#    plt.text(a, b, str(b))
plt.savefig("M_REAL")


f=open("M_ALL_SYNTH.txt","r")
x=[]
y1=[]
y2=[]
y3=[]
for line in f:
    X=line.split()
    x.append(float(X[0]))
    y1.append(float(X[1]))
    y2.append(float(X[2]))
    y3.append(float(X[3]))

plt.figure(2)
plt.title("M - Synthetic Dataset")
plt.xlabel("M")
plt.ylabel("ERMS")
a,=plt.plot(x,y1,label="Train")
b,=plt.plot(x,y2,label="Validation")
c,=plt.plot(x,y3,label="Testing")
plt.legend(handles=[a,b,c])
plt.savefig("M_Synthetic")


f=open("GRAD_REAL.txt","r")
x=[]
y1=[]
y2=[]
y3=[]
for line in f:
    X=line.split()
    x.append(float(X[0]))
    y1.append(float(X[1]))
    y2.append(float(X[2]))
    y3.append(float(X[3]))

plt.figure(3)
plt.title("Gradient Descent - LeToR dataset")
plt.xlabel("Iterations")
plt.ylabel("ERMS")
a,=plt.plot(x,y1,label="Train")
b,=plt.plot(x,y2,label="Validation")
c,=plt.plot(x,y3,label="Testing")
plt.legend(handles=[a,b,c])
plt.savefig("Grad_real")



f=open("GRAD_SYNTH.txt","r")
x=[]
y1=[]
y2=[]
y3=[]
for line in f:
    X=line.split()
    x.append(float(X[0]))
    y1.append(float(X[1]))
    y2.append(float(X[2]))
    y3.append(float(X[3]))

plt.figure(4)
plt.title("Gradient Descent - Synthetic dataset")
plt.xlabel("Iterations")
plt.ylabel("ERMS")
a,=plt.plot(x,y1,label="Train")
b,=plt.plot(x,y2,label="Validation")
c,=plt.plot(x,y3,label="Testing")
plt.legend(handles=[a,b,c])
plt.savefig("Grad_Synth")

plt.show()




f=open("ld.txt","r")
x=[]
y1=[]
for line in f:
    X=line.split()
    x.append(float(X[0]))
    y1.append(float(X[1]))
    
plt.figure(5)
plt.title("Lambda LeToR Dataset")
plt.xlabel("Lambda")
plt.ylabel("ERMS")
a,=plt.plot(x,y1,label="Validation")
plt.legend(handles=[a])
plt.savefig("Lambda_real")

plt.show()



f=open("ld_synth.txt","r")
x=[]
y1=[]
for line in f:
    X=line.split()
    x.append(float(X[0]))
    y1.append(float(X[1]))
    
plt.figure(6)
plt.title("Lambda Synthetic Dataset")
plt.xlabel("Lambda")
plt.ylabel("ERMS")
a,=plt.plot(x,y1,label="Validation")
plt.legend(handles=[a])
plt.savefig("lambda_synth")
plt.show()

f=open("alpha_real.txt","r")
x=[]
y1=[]
for line in f:
    X=line.split()
    x.append(float(X[0]))
    y1.append(float(X[1]))


plt.figure(7)
plt.title("Learning Rate LeToR Dataset")
plt.xlabel("Iterations")
plt.ylabel("Learning Rate")
plt.plot(x,y1)
plt.savefig("alpha_real")
plt.show()


f=open("alpha_synth.txt","r")
x=[]
y1=[]
for line in f:
    X=line.split()
    x.append(float(X[0]))
    y1.append(float(X[1]))


plt.figure(7)
plt.title("Learning Rate Sytnthetic Dataset")
plt.xlabel("Iterations")
plt.ylabel("Learning Rate")
plt.savefig("alpha_synth")
plt.plot(x,y1)

plt.show()

