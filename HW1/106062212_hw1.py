#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
from bs4 import BeautifulSoup
f = open("input_hw1.txt")
lines = f.readlines()
output = open("106062212_hw1_output.txt",'w+', encoding='UTF-8') 
for line in lines:
    time = 3
    add = line
    route = add.strip()
    while(time>=0):
        addr = "https://www.blockchain.com/eth/address/" + add.strip() + "?view=standard"
        if time<3:
            r = " -> "+ add.strip()
            route+=r
        r = requests.get(addr)
        soup = BeautifulSoup(r.text,"html.parser")
        spans = soup.find("div", class_ = 'hnfgic-0 blXlQu').find_all('span')
        for i in range(2, len(spans), 2):
            attr = spans[i].text.strip()+': '+spans[i+1].text.strip()
            print(attr, file = output)
        given_id = []
        given_date = []
        given_amount = []
        directs = soup.find_all("div", class_ = 'sc-1fp9csv-0 gkLWFf')
        for direct in directs:
            given_date.append(direct.find("span", class_ = 'sc-1ryi78w-0 bFGdFC sc-16b9dsl-1 iIOvXh u3ufsr-0 gXDEBk').text.strip())
            given = direct.find_all("a", class_ = 'sc-1r996ns-0 dEMBMB sc-1tbyx6t-1 gzmhhS iklhnl-0 dVkJbV')
            given_id.append(given[2].text.strip())
            try:
                given_amount.append(direct.find("span", class_ = 'sc-1ryi78w-0 bFGdFC sc-16b9dsl-1 iIOvXh u3ufsr-0 gXDEBk sc-85fclk-0 gskKpd').text.strip()) 
            except:
                given_amount.append(' ')
        tar = len(given_amount)-1
        while given_amount[tar]==" ":
            tar-=1
            if tar==-1:
                print("--------------------------------------------------------------------------", file = output)
                break
        if (tar==-1):
            break
        else:
            time-=1
            add = given_id[tar]
        print ("Date: " + given_date[tar], file = output)
        print("To: " + given_id[tar], file = output)
        print("Amount: " + given_amount[tar], file = output)
        print("--------------------------------------------------------------------------", file = output)
    print(route, file = output)
    print("--------------------------------------------------------------------------", file = output)
output.close()        
f.close()


# In[ ]:




