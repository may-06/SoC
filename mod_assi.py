import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df=pd.read_csv("train.csv")



#print(df)
df_train=df.iloc[1001:]
X_dev=df_train.drop(columns="label")
X_dev=X_dev.to_numpy().T
X_dev=X_dev/255.00
#print(X_dev.shape)

Y_dev=df_train["label"]
Y_dev=Y_dev.to_numpy().T
#print(Y_dev.size)

#plt.imshow(X_dev[:,80].reshape(28,28))


df_test=df.iloc[:1000]
X_test=df_test.drop(columns="label")
X_test=X_test.to_numpy().T
X_test=X_test/255.00
#print(X_test.shape)


Y_test=df_test["label"].to_numpy()
Y_test=np.transpose(Y_test)



def init_params():
  W1=np.random.rand(120,784)-0.5
  #print(W1)
  b1=np.random.rand(120,1)-0.5
  W2=np.random.rand(45,120)-0.5
  b2=np.random.rand(45,1)-0.5
  W3=np.random.rand(10,45)-0.5
  b3=np.random.rand(10,1)-0.5
  return (W1,b1,W2,b2,W3,b3)



def ReLU(z):
    return np.maximum(0,z)
 
def softmax(z):
    #print(z.shape)
    softmax=np.zeros(z.shape)
    #print(softmax.shape)
    for i in range(z.shape[1]):
         z1=z[:,i]
         x = z1 - max(z1)
         num = np.exp(x)
         denom = np.sum(num)
         softmax[:,i]=num/denom

    return softmax

def forward_propagation(X,W1,b1,W2,b2,W3,b3):
    #print(W1.shape,"hi")
    Z1=np.dot(W1,X) + b1
    #print(Z1[:,80],"\n\n")
    A1=ReLU(Z1)
    #print(A1[:,80],"\n\n")
    
    Z2=np.dot(W2,A1) + b2
    #print(Z2[:,80],"\n\n")
    A2=ReLU(Z2)
    #print(A2[:,80],"\n\n")
    
    Z3=np.dot(W3,A2) + b3
    #print(Z3[:,80])
    A3=softmax(Z3)
    #print(A3.shape)
    #print(A3[:,80])
    
    return (Z1,A1,Z2,A2,Z3,A3)


   
def one_hot(Y):
    l=[]
    for i in Y.reshape(Y.size):
         #print(i)
         b=np.zeros(10)
         b[i]=1
         #print(b)
         l.append(b)
         #a[i]=b
    Y=np.array(l)
    Y=np.transpose(Y)
    return Y



#print(Y_test.shape)
    
def backward_propagation(Z1,A1,Z2,A2,Z3,A3,W1,W2,W3,X,Y):
    dZ3 = A3 - Y
    dW3 = np.dot(dZ3, A2.T)/41000.0
    db3 = np.sum(dZ3, axis=1, keepdims=True)/41000.0
    
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int32(A2 > 0))#elementwise
    dW2 = np.dot(dZ2, A1.T)/41000.0
    db2 = np.sum(dZ2, axis=1, keepdims=True)/41000.0
    
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int32(A1 > 0))
    dW1 = np.dot(dZ1, X.T)/41000.0
    db1 = np.sum(dZ1, axis=1, keepdims=True)/41000.0
    
    return dW1,db1,dW2,db2,dW3,db3
    
def update_params(W1,b1,W2,b2,W3,b3,dW1,db1,dW2,db2,dW3,db3,alpha):
    W3=W3-alpha*dW3
    b3=b3-alpha*db3
    W2=W2-alpha*dW2
    b2=b2-alpha*db2
    W1=W1-alpha*dW1
    b1=b1-alpha*db1

    return(W1,b1,W2,b2,W3,b3)

def get_prediction(A):
    return np.argmax(A,axis=0)

def get_accuracy(y,Y):
    #print(y.shape)
    x=np.sum(y==Y)
    acc=x/y.shape
    return acc*100


def gradient_descent(X_dev,Y_dev,alpha,num):
    W1, b1, W2, b2, W3, b3 = init_params()
    Y_one_hot = one_hot(Y_dev)
    
    for i in range(num):
        #print("hiiiii")
        Z1,A1,Z2,A2,Z3,A3 = forward_propagation(X_dev,W1,b1,W2,b2,W3,b3)
        dW1,db1,dW2,db2,dW3,db3 = backward_propagation(Z1,A1,Z2,A2,Z3,A3,W1,W2,W3,X_dev,Y_one_hot)
        W1,b1,W2,b2,W3,b3 = update_params(W1,b1,W2,b2,W3,b3,dW1,db1,dW2,db2,dW3,db3,alpha)
        
        if (i + 1) % 10 == 0:
            pred = get_prediction(A3)
            print("output layer:(every 10th iteration ", A3,"\n")
            acc = get_accuracy(pred, Y_dev)
            print("Accuracy is ",acc,"\n\n")
    
    return (W1,b1,W2,b2,W3,b3)

def make_predictions(X,W1,b1,W2,b2,W3,b3):    
    A3=forward_propagation(X,W1,b1,W2,b2,W3,b3)[-1]
    pred=get_prediction(A3)
    return pred




def test_prediction(index,X,Y,W1,b1,W2,b2,W3,b3):
    prediction = make_predictions(X[:,index].reshape(-1,1),W1,b1,W2,b2,W3,b3)
    true = Y[index]
    img= X[:,index].reshape(28,28)

    plt.imshow(img)
    plt.title(f"Prediction: {prediction}, True Label: {true}")
    plt.show()
    
        
W1,b1,W2,b2,W3,b3=gradient_descent(X_dev,Y_dev,0.1,1000)    

test_prediction(300,X_test,Y_test,W1,b1,W2,b2,W3,b3)
    
    
    
    
    
    
    
    
    
    