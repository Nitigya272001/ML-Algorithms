import pandas as pd
from matplotlib import pyplot as plt
import math

def get_graph(ks,val):
    plt.bar(ks,val)
    plt.xlabel("Different set of K used")
    plt.ylabel("Accuracy")
    plt.title("Graph between Accuracy and K-fold")
    plt.show()

def make_split(leng,idx,partLength):
    train=[]
    cross=[]
    for i in range((idx-1)*partLength):
        if i==0:
            continue
        train.append(i)

    for i in range((idx-1)*partLength,idx*partLength):
        if i==0:
            continue
        cross.append(i)
        
    for i in range((idx*partLength),leng):
        if i==0:
            continue
        train.append(i)
        
    return cross,train    

def get_euclid(x,y,len):
    ans = 0
    
    for i in range(len):
        ans = ans + (x[i]-y[i])**2
        
    return math.sqrt(ans)

def get_prediction(data,cross,train,k):
    result={}
    correct=0
    for i in range(len(cross)):
        x1=data['Total_Bilirubin'][cross[i]]
        x2=data['Direct_Bilirubin'][cross[i]]
        x3=data['Alkaline_Phosphotase'][cross[i]]
        x4=data['Alamine_Aminotransferase'][cross[i]]
        x5=data['Aspartate_Aminotransferase'][cross[i]]
        x6=data['Total_Protiens'][cross[i]]
        x7=data['Albumin'][cross[i]]
        x8=data['Albumin_and_Globulin_Ratio'][cross[i]]
        x = [x1,x2,x3,x4,x5,x6,x7,x8]
        
        for j in range(len(train)):
            y1=data['Total_Bilirubin'][train[j]]
            y2=data['Direct_Bilirubin'][train[j]]
            y3=data['Alkaline_Phosphotase'][train[j]]
            y4=data['Alamine_Aminotransferase'][train[j]]
            y5=data['Aspartate_Aminotransferase'][train[j]]
            y6=data['Total_Protiens'][train[j]]
            y7=data['Albumin'][train[j]]
            y8=data['Albumin_and_Globulin_Ratio'][train[j]]
            y = [y1,y2,y3,y4,y5,y6,y7,y8]
            
            ans = get_euclid(x,y,8)
            result[ans]=data['Dataset'][train[j]]
    
        c1 = 0
        c2 = 0
        count = 0
        
        for z in sorted(result):
            if(count == k): break
            
            if(result[z]==1): c1=c1+1
            
            elif(result[z]==2): c2=c2+1
            
            count = count + 1
        
        predicted=max(c1,c2)
        
        if(predicted==c1):
            if data['Dataset'][cross[i]] == 1 :
                correct =  correct+1
        else:
            if(data['Dataset'][cross[i]]==2):
                correct =  correct+1
        
        result.clear()
    
    return correct  
      
data = pd.read_csv(r'C:\Users\acer\Desktop\SEM4\ML\Lab4\liver_total_train.csv')
leng = data['Total_Bilirubin'].count()
K = 231

k_set = [2,5,10]

for k in k_set:
    partLength=int(leng/k)
    avg_acc=0
    mx_acc=0
    part = {}
    result=[]
    for i in range(1,k+1):
        cross,train = make_split(leng,i,partLength)
        temp = get_prediction(data,cross,train,K)    
        temp = temp/(len(cross))*100                    
        mx_acc=max(mx_acc,temp)
        avg_acc=avg_acc+temp
        part[i]=temp
    
        if(temp==avg_acc):
            result.clear()
            result=cross

    get_graph(list(part.keys()),list(part.values()))
    print('For k-fold and k : ',k)
    print("Maximum accuracy  : ",mx_acc)
    print("Average accuracy : ",(avg_acc/(k*100))*100)