#!/usr/bin/env python
# coding: utf-8

# In[1]:


from itertools import combinations
import sys


# In[2]:


class node:
    def __init__(self, product, parent):
        self.product = product
        self.count = 1
        self.left = None
        self.parent = parent
        self.children = {}


# In[3]:


class next_node:
    def __init__(self, node):
        self.FP_node = node
        self.next = None

# In[1]:

min_support = float(sys.argv[1])
input_file = sys.argv[2]
output_file = sys.argv[3]
fp = open(output_file, "w+")
f = open(input_file)
lines = f.readlines()
header_array = [0] * 1000
header_table = []
count = 0
new_array = []
for line in lines:
    line = line.strip()
    line = line.split(' ')
    new_array.append([])
    for item in line:
        header_array[int(item)] += 1
        new_array[count].append(int(item))
    count += 1
for i in range(1000):
    if(header_array[i] != 0):
        header_table.append((i,header_array[i]))
header_table.sort(key=lambda header_table: header_table[1], reverse = True)
count_final = count
count *= min_support

stop_index = 0
for i in range(len(header_table)):
    if(header_table[i][1] < count):
        break
    stop_index += 1
    
    
FP_root = node('root', None)
header_tree = {}

for line in new_array:
    #line = line.strip()
    #line = line.split(',')
    current_node = FP_root
    for i in range(stop_index):
        if header_table[i][0] in line:
            if header_table[i][0] in current_node.children:
                current_node = current_node.children[header_table[i][0]]
                current_node.count += 1
                
            else:
                tmp = current_node
                c_node = node(header_table[i][0] , tmp)
                current_node.children[header_table[i][0]] = c_node
                current_node = current_node.children[header_table[i][0]]
                
                if header_table[i][0] not in header_tree:
                    n_node = next_node(current_node)
                    header_tree[header_table[i][0]] = n_node
                else:
                    header_now = header_tree[header_table[i][0]] 
                    while(header_now.next != None):
                        header_now = header_now.next
                    n_node = next_node(current_node)
                    header_now.next = n_node
final_answer = []        

for i in range(stop_index):
    cd_tree = []
    sum_array = {}
    key_array = []
    now2 = header_tree[header_table[i][0]]
    while(now2 != None):
        tmp = now2.FP_node.count
        now = now2.FP_node
        while(now.parent.product != 'root'):
            now = now.parent
            if now.product in sum_array:
                sum_array[now.product] += tmp
            else:
                sum_array[now.product] = tmp
        now2 = now2.next
    for key in sum_array.keys():
        if(sum_array[key] >= count):
            key_array.append(key)

    now2 = header_tree[header_table[i][0]]
    while(now2 != None):
        rule_array = []
        tmp = now2.FP_node.count
        now = now2.FP_node
        while(now.parent.product != 'root'):
            now = now.parent
            if now.product in key_array:
                rule_array.append(now.product)
        if(rule_array != []):
            for z in range(tmp):
                cd_tree.append((rule_array))
        now2 = now2.next

    index_list = []

    CD_root = node('root', None) 
    header_tree_tmp = {}
    for cd_i in range(len(cd_tree)):
        b = len(cd_tree[cd_i])
        current_node = CD_root
        for cd_j in range(b):
            cd_node = cd_tree[cd_i][b-cd_j-1]
            if cd_node not in index_list:
                index_list.append(cd_node)
            if cd_node in current_node.children:
                current_node = current_node.children[cd_node]
                current_node.count += 1
            else:
                tmp = current_node
                c_node = node(cd_node , tmp)
                current_node.children[cd_node] = c_node
                current_node = current_node.children[cd_node]

                if cd_node not in header_tree_tmp:
                    n_node = next_node(current_node)
                    header_tree_tmp[cd_node] = n_node
                else:
                    header_now = header_tree_tmp[cd_node] 
                    while(header_now.next != None):
                        header_now = header_now.next
                    n_node = next_node(current_node)
                    header_now.next = n_node

    for item in index_list:
        sum_array = {}
        key_array = []
        now2 = header_tree_tmp[item]
        while(now2 != None):
            tmp = now2.FP_node.count
            now = now2.FP_node
            while(now.parent.product != 'root'):
                now = now.parent
                if now.product in sum_array:
                    sum_array[now.product] += tmp
                else:
                    sum_array[now.product] = tmp
            now2 = now2.next
        for key in sum_array.keys():
            if(sum_array[key] >= count):
                key_array.append(key)

        answer = []
        now2 = header_tree_tmp[item]
        while(now2 != None):
            rule_array = []
            tmp = now2.FP_node.count
            now = now2.FP_node
            while(now.parent.product != 'root'):
                now = now.parent
                if now.product in key_array:
                    rule_array.append(now.product)
            if(rule_array != []):
                rule_array.sort()
                for com_i in range(len(rule_array)+1):
                    if com_i ==1:
                        com_item = []
                        com_item.append(item)
                        com_item.append(header_table[i][0])
                        com_item.sort()
                        if answer == []:
                            answer.append([com_item,tmp])
                        else:
                            final_flag = 0
                            for final_i in range(len(answer)):
                                if com_item == answer[final_i][0]:
                                    answer[final_i][1] += tmp   
                                    final_flag = 1
                            if(final_flag == 0):
                                answer.append([com_item,tmp])
                    if com_i >= 1:
                        com_rule_list = combinations(rule_array,com_i)
                        for com_item in com_rule_list:
                            com_item = list(com_item)
                            com_item.append(item)
                            com_item.append(header_table[i][0])
                            com_item.sort()
                            if answer == []:
                                answer.append([com_item,tmp])
                            else:
                                final_flag = 0
                                for final_i in range(len(answer)):
                                    if com_item == answer[final_i][0]:
                                        answer[final_i][1] += tmp   
                                        final_flag = 1
                                if(final_flag == 0):
                                    answer.append([com_item,tmp])
            else:
                rule_array.append(item)
                rule_array.append(header_table[i][0])
                rule_array.sort()
                if answer == []:
                    answer.append([rule_array,tmp])
                else:
                    final_flag = 0
                    for final_i in range(len(answer)):
                        if rule_array == answer[final_i][0]:
                            answer[final_i][1] += tmp   
                            final_flag = 1
                    if(final_flag == 0):
                        answer.append([rule_array,tmp])

            now2 = now2.next
        for item in answer:
            final_answer.append(item)
    



final_answer.sort(key=lambda final_answer: final_answer[0])
final_answer.sort(key=lambda final_answer: len(final_answer[0]))
header_table.sort(key=lambda header_table: header_table[0])
for item in header_table:
    if(item[1] >= count):
        a = item[1]/count_final
        print(item[0],":", '%.4f'%a,sep="",file = fp)
for item in final_answer:
    for i in range(len(item[0])-1):
        print(item[0][i],",",end = "",sep="",file = fp)
    print(item[0][len(item[0])-1],":",end = "",sep="",file = fp) 
    a = item[1]/count_final
    print('%.4f'%a,sep="",file = fp)
fp.close()
f.close() 




