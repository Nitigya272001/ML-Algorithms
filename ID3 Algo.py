
##### AIM - Use ID3 algorithm and k-fold cross validation to classify the bank dataset approving loan or not.

import pandas as pd
import numpy as np
import random
eps = np.finfo(float).eps

headings = []


def load_datasets(S,folds):
    test_set=[]
    training_set=[]
    for row in S:
        if random.random()<1/(folds+1):
            test_set.append(row)
        else:
            training_set.append(row)
    
    return training_set,test_set


def unique_values(S,feature_idx):
    feature_values={}
    target_values={}
    for row in S:
        if row[feature_idx] not in feature_values:
            feature_values[row[feature_idx]] = {'yes':0,'no':0}

        if(row[-1]) not in target_values:
            target_values[row[-1]]=0
        
        target_values[row[-1]] += 1
        feature_values[row[feature_idx]][row[-1]] += 1
    
    
    return feature_values,target_values


def Entropy(S):
      S_yes,S_no = 0,0

      for row in S:
            if row[-1] == 'yes':
                  S_yes = S_yes + 1

            elif row[-1] == 'no':
                  S_no = S_no + 1

      if S_yes == 0 or S_no == 0:
            return 0

      elif S_no == S_yes:
            return 1

      else:
            p_yes = S_yes/(S_yes + S_no)
            p_no = S_no/(S_no + S_yes)

            Entropy = p_yes*np.log2(p_yes) + p_no*np.log2(p_no)
            return Entropy*(-1)

def IG(S,feature_idx):
      total_entropy = Entropy(S)
      feature_values,labels = unique_values(S,feature_idx)
      total = len(S)
      gain = 0
      for feature_value, tar_val in feature_values.items():
            temp_total = tar_val['yes'] +tar_val['no']

            for_yes = tar_val['yes']/temp_total*np.log2(tar_val['yes']+eps/ temp_total)
            for_no = tar_val['no']/temp_total*np.log2(tar_val['no'] +eps/ temp_total)
            
            feature_entropy= for_yes + for_no
            
            gain = gain + (temp_total/ total) * (-feature_entropy)
      
      return total_entropy-(gain),labels,feature_values


def IG_max(S,feature_list):
    ig_max=-1000000
    final_l={}
    final_d={}
    final_col=0

    for feature_idx in feature_list:
        ig,target_values,feature_values = IG(S,feature_idx)
        
        if(ig>ig_max):
            ig_max=ig
            final_l=target_values
            final_d=feature_values
            final_col=feature_idx

    return final_l,ig_max,final_d,final_col

def build_tree(S,feature_list,a):
    labels,info,dic,col=IG_max(S,feature_list)
    
    if len(labels)==1:
        return list(labels.keys())[0]
    
    if len(feature_list)==1:
        mx='yes'
        t=0
        for tar,cnt in labels.items():
            if t<=cnt:
               mx=tar
               t=cnt
        return mx

    split_sets={}
    feature_values=list(dic.keys())
    for value in feature_values:
        split_sets[value]=[]

    for row in S:
        split_sets[row[col]].append(row)

    feature_list.remove(col)
    new_feature_list = feature_list.copy()
    final_tree={headings[col]:{}}
    
    for val,rows in split_sets.items():
        final_tree[headings[col]][val]= build_tree(rows,new_feature_list,a+1)
    
    return final_tree 

def check(tree,inst):
    if(type(tree)==str):
        return tree
    col=list(tree.keys())[0]
    col_no=head_dic[col]
    if inst[col_no] not in tree[col]:
        return
    t=check(tree[col][inst[col_no]],inst)
    return t

def cal_accuracy(tree,test_set):
    total=0
    correct=0
    for inst in test_set:
        t=check(tree,inst)
        if t==inst[-1]:
            correct+=1
        if type(t)==str:
            total+=1

    accuracy=(correct/total)*100
    return accuracy

def k_folds(S):
    acc=0
    final_tree={}
    for i in range(2,6):
        training_set, test_set = load_datasets(S,3)
        tree = build_tree(training_set, [k for k in range(len(S[0]) - 1)], 0)
        temp=cal_accuracy(tree, test_set)
        if temp>acc:
            acc=temp
            final_tree=tree

        print('folds :',i,'=> accuracy : ',temp)

    return acc


df = pd.read_csv(r"ML\Lab8\bank.csv")
S = [df.columns.values.tolist()] + df.values.tolist()

headings= [x for x in S[0]]

head_dic={}
for i in range(len(headings)):
      head_dic[headings[i]] = i

S.pop(0)

print('maximum accuracy :',k_folds(S))