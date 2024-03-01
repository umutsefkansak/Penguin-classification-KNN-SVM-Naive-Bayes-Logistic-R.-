import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%%

df = pd.read_csv("penguins_classification.csv")

#%%

df.info()

print(df.isna().sum())
print("Total null values = ",df.isnull().sum().sum())


#%%

df.drop(["island"],axis=1,inplace = True)

#%%

Adelie = df[df.species == "Adelie"]
Gentoo = df[df.species == "Gentoo"]

#%%

plt.scatter(Adelie.bill_length_mm,Adelie.bill_depth_mm,color = "red",label = "Adelie",alpha = 0.7)
plt.scatter(Gentoo.bill_length_mm,Gentoo.bill_depth_mm,color = "green",label = "Gentoo",alpha = 0.7)
plt.xlabel("bill length mm")
plt.ylabel("bill depth mm")
plt.legend()
plt.show()

#%%

df.species = [1 if each == "Adelie" else 0 for each in df.species]

#%%

x_data = df.drop(["species"],axis = 1)
y = df.species.values


#%%



x = (x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))
#%%


from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x, y,test_size=0.2,random_state=42)


#%%

from sklearn.neighbors import KNeighborsClassifier

score_list = []
for i in range(1,15):
    knn2 = KNeighborsClassifier(n_neighbors=i)
    knn2.fit(x_train,y_train)
    score_list.append(knn2.score(x_test.values,y_test))


plt.plot(range(1,15),score_list)
plt.show()

#%%
best_k = 0

for i in range(len(score_list)):
    
    if score_list[i] > score_list[best_k]:
        best_k = i
        

best_k = best_k+1 # because score_list[0] k=1, score_list[1] k=2,..., score_list[n] k=n+1


#%%

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(x_train,y_train)

print("{} nn score = {} ".format(best_k,knn.score(x_test.values, y_test)))

#%%

from sklearn.svm import SVC

svm = SVC()

svm.fit(x_train,y_train)
print("Svm score: ",svm.score(x_test, y_test))



#%%

from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()

nb.fit(x_train,y_train)

print("naive bayes score = ",nb.score(x_test, y_test))



#%% 

# Logistic Regression
x_train = x_train.T
x_test = x_test.T
y_train = y_train.T
y_test = y_test.T

#%%

def init(dimension):
    w = np.full((dimension,1),0.01)
    b = 0.0
    return w,b

#%%
def sigmoid(z):
    y_head = 1/(1+np.exp(-z))
    return y_head

#%%

def forward_backward_propagation(w,b,x_train,y_train):
    
    z = np.dot(w.T,x_train) + b
    y_head = sigmoid(z)
    loss = -y_train*np.log(y_head)-(1-y_train)*np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    
    #Backward propagation
    derivative_weight = (np.dot(x_train,((y_head-y_train).T)))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    
    gradients = {"derivative_weight":derivative_weight,"derivative_bias":derivative_bias}
    
    return cost,gradients


#%%


def update(w,b,x_train,y_train,learning_rate,number_of_iteration):
    
    for i in range(number_of_iteration):
        
        cost,gradients = forward_backward_propagation(w, b, x_train, y_train)
        
        w = w - learning_rate*gradients["derivative_weight"]
        b = b - learning_rate*gradients["derivative_bias"]
        
        if i % 20 == 0:
            print(i,"->iteration cost = ",cost)
        
    parameters = {"weight":w,"bias":b}
    
    return parameters

#%%


def prediction(w,b,x_test):
    
    y_heads = sigmoid(np.dot(w.T,x_test)+b)
    y_predictions = np.zeros((1,x_test.shape[1]))
    
    
    
    for i in range(y_heads.shape[1]):
        
        if y_heads[0,i] <= 0.5:
            y_predictions[0,i] = 0
        else:
            y_predictions[0,i] = 1
    
    
    return y_predictions



#%%


def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,number_of_iteration):
    
    
    dimension = x_train.shape[0]
    w,b = init(dimension)
    
    parameters = update(w, b, x_train, y_train, learning_rate, number_of_iteration)
    
    
    y_predictions = prediction(parameters["weight"], parameters["bias"], x_test)
    
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_predictions - y_test)) * 100))





#%%

logistic_regression(x_train, y_train, x_test, y_test, 1, 200)





























