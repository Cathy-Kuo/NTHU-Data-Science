#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
from itertools import combinations


# In[45]:


min_support = float(sys.argv[1])
in_txt = sys.argv[2]
out_txt = sys.argv[3]
f = open(in_txt)
lines = f.readlines()
data = [0]*1000
header_table = []
header_link = []
check = []
total = len(lines)
min_support = min_support*len(lines)
output = open(out_txt,'w+')
for i in range(len(lines)):
    lines[i] = lines[i].strip().split(' ')
    for j in range(len(lines[i])):
        lines[i][j] = int(lines[i][j])
        data[lines[i][j]]+=1
for i in range(1000):
    header_link.append([])
    if data[i]>=min_support:
        header_table.append((i, data[i]))
header_table.sort(key=lambda header_table: header_table[1], reverse=True)
for item in header_table:
    check.append(item[0])
DB = []
for i in range(len(lines)):
    DB.append([])
    for item in check:
        if item in lines[i]:
            DB[i].append(item)


# In[46]:


class treeNode:
    def __init__(self, nameValue, numOccur, parentNode):
        self.name = nameValue
        self.count = numOccur
        self.nodeLink = None
        self.parent = parentNode
        self.children = {}


# In[47]:


class linkNode:
    def __init__(self, Node):
        self.node = Node
        self.next = None


# In[48]:


def createFPtree(min_support):
    fpTree = treeNode('Root', 1, None)
    for line in DB:
        pre_node = fpTree
        for item in line:
            if item in pre_node.children:
                pre_node.children[item].count+=1
            else:
                pre_node.children[item] = treeNode(item, 1, pre_node)
                header_link[item].append(pre_node.children[item])
            pre_node = pre_node.children[item]
    return fpTree


# In[49]:


def Combine(freq_pat, suf_count, oriSuf, suffix):
    num = len(freq_pat)
    freq = []
    l = [oriSuf,suffix]
    l.sort()
    freq.append([suf_count, l])
    for i in range(1, num+1):
        combList = list(combinations(freq_pat, i))
        for j in range(len(combList)):
            combList[j] = list(combList[j])
            combList[j].append(oriSuf)
            combList[j].append(suffix)
            combList[j].sort()
            freq.append([suf_count, combList[j]])
    return freq


# In[50]:


def conditionalTree(oriSuf, condiDB):
    condiTree = treeNode('Root', 1, None)
    freq_pat = []
    condiHeaderLink = {}
    headerTable = []
    freq = []
    for line in condiDB:
        l = len(line)
        pre_node = condiTree
        if l!=1:
            cnt = line[0]
            for i in range(l-1, 0, -1):
                if line[i] not in headerTable:
                    headerTable.append(line[i])
                if line[i] in pre_node.children:
                    pre_node.children[line[i]].count+=cnt
                else:
                    pre_node.children[line[i]] = treeNode(line[i], cnt, pre_node)
                    if line[i] not in condiHeaderLink:
                        condiHeaderLink[line[i]] = linkNode(pre_node.children[line[i]])
                    else:
                        now = condiHeaderLink[line[i]]
                        while now.next!=None:
                            now = now.next
                        now.next = linkNode(pre_node.children[line[i]])
                pre_node = pre_node.children[line[i]]
        
    num_item = len(headerTable)
    j=0
    for suffix in headerTable:
        count = {}
        first = condiHeaderLink[suffix]
        link = condiHeaderLink[suffix]
        startNode = condiHeaderLink[suffix]
        while link!=None:
            freq_pat.append([])
            node = link.node
            suf_count = node.count
            if int(node.name) not in count:
                count[int(node.name)]=suf_count
            else:
                count[int(node.name)]+=suf_count

            while node.parent.name!='Root':
                if int(node.parent.name) not in count:
                    count[int(node.parent.name)]=suf_count
                else:
                    count[int(node.parent.name)]+=suf_count
                node = node.parent
            link = link.next
        
        while startNode!=None:
            node = startNode.node
            suf_count = node.count
            while node.parent.name!='Root':
                if count[int(node.parent.name)]>=min_support:
                    freq_pat[j].append(node.parent.name)
                node = node.parent
            list1 = Combine(freq_pat[j], suf_count, oriSuf, startNode.node.name)
            for item in list1:
                freq.append(item)
            startNode = startNode.next
            j+=1
    return freq


# In[51]:


fpTree = createFPtree(min_support)
num_item = len(header_table)
frequent = []
for i in range(num_item-1, -1, -1):
    condiDB = []
    j = 0
    count = [0]*1000
    suffix = header_table[i][0]
    for node in header_link[suffix]:
        suf_count = node.count
        condiDB.append([suf_count])
        count[int(node.name)]+=suf_count
        while node.parent.name!='Root':
            count[int(node.parent.name)]+=suf_count
            node = node.parent
    for node in header_link[suffix]:
        node = node.parent
        while node.name!='Root':
            if count[int(node.name)]>=min_support:
                condiDB[j].append(node.name)
            node = node.parent
        j+=1
    condiList = conditionalTree(suffix, condiDB)
    if len(condiList)!=0:
        for item in condiList:
            flag = 0
            for fr in frequent:
                if item[1]==fr[1]:
                    fr[0]+=item[0]
                    flag = 1
            if flag==0:
                frequent.append([item[0], item[1]])

# In[52]:


frequent.sort(key = lambda frequent:frequent[1])
frequent.sort(key = lambda frequent:len(frequent[1]))
header_table.sort()
for item in header_table:
    F = item[1]/total
    print(item[0],':','%.4f'%F, sep='', file=output)
for item in frequent:
    F = round(item[0]/total, 4)
    print(",".join(str(i) for i in item[1]),':','%.4f'%F, sep='', file=output) 
output.close()  
f.close()


# In[ ]:




