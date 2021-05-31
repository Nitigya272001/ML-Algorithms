import math
import random
from matplotlib import pyplot as plt
from random import randint
#import csv



def read_dataset(file_name,split,training_set=[],test_set=[]):
    
    with open(file_name,'r') as my_file:
        lines=my_file.readlines()         #it read each line as tupple from file
         
        for i in range(1,len(lines)-1):   #skipping first line because it
                                          #contains column names not data
                                          
             temp=lines[i].split(',')      #tuple unpacking 
             for j in range(len(temp)-1):
                 temp[j]=float(temp[j])    #converting data type to float because 
                                           #by default data type is str and we can't
                                           #perform mathmatically operation on str
               
             if (random.random()<split):
                training_set.append(temp)     
             else:
                test_set.append(temp)
            

                
                
def euclid_distance(point1,point2,length) :
    distance=0
    
    for i in range(length) :
        distance+=(point1[i]-point2[i])**2    #calculating eculian distance
        
    return math.sqrt(distance)


     
        
def get_nieghbours(sample_data,training_set,k):
    
    neighbours_distance=[]
    
    length=4;
    
    dist=0;
    for i in range(len(training_set)):
        
        dist=euclid_distance(sample_data,training_set[i],length)
        neighbours_distance.append((dist,training_set[i]))
        
    neighbours_distance.sort()             #sorting data according to distance
                                           # in ascending order
    neighbours=[]
    
    for i in range(k):
        neighbours.append(neighbours_distance[i][1])
        
    return neighbours
  
      
 
def get_prediction(neighbours):
    visited={}                  # to count frequency
    for x in neighbours:
        flower=x[-1]
        if flower in visited:
            visited[flower]+=1
        else:
            visited[flower]=1
    
    
    visited=dict(sorted(visited.items(),key=lambda item:item[1],reverse=True))
   
    #sorting the dictionary in descending order by value
    return (list(visited.keys())[0])
   
    
def get_accuracy(correct,total):
    acc=round((correct/total)*100,2);
    print(f"Acurracy of this model(algo) is :{acc}")
    return acc
    
    
    
def get_graph(accuracy,diff_k):
    plt.plot(diff_k,accuracy)
    plt.legend()
    plt.title("Graph Between Accuracy and K")
    plt.ylabel("Accuracy")
    plt.xlabel("value of K")
    plt.show()
           

def print_result(result):
    for x in result:
        for y in x:
            print(y,end="      ")
        print("\n")


def main():
   
    training_set=[]
    test_set=[]
        
    read_dataset(r'ML\Lab3\iris.csv',0.70,training_set,test_set)
    total=len(test_set)
    accuracy=[]
    diff_k=[]
    my_dict={}
    for z in range(1,50):
        
        k=int(math.sqrt(len(training_set))-7+randint(1,50))
        if(k%2==0):
            k=k-1
            
        if k in diff_k:
            continue
        correct=0;
        
        for x in test_set:
            
            neighbours=get_nieghbours(x, training_set, k)
            result=get_prediction(neighbours);
            if(result==x[-1]):
                correct+=1;
        print(f"\n\ntraining_set Size : {len(training_set)}")
        print(f"test_set Size : {len(test_set)}")
        my_dict[k]=get_accuracy(correct, total)
    
    my_dict=dict(sorted(my_dict.items(),key=lambda item:item[0]))
    diff_k=list(my_dict.keys())
    accuracy=list(my_dict.values());
    get_graph(accuracy,diff_k)
    
    print(f"\naccuracy are : {accuracy}")
    print(f"corresponding k values : {diff_k}")

main()