# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 22:53:01 2024

@author: LENOVO
"""

#decision tree
import numpy as np 
import pandas as pd


data=pd.read_csv("toy_dataset_decision_tree.csv")


yes=0
no=0
for values in data["Class (Play Tennis)"]:
    if values=="Yes":
        yes+=1
    else:
        no+=1


def entropi(x, y):
    total = x + y
    if total != 0:
        p_x = x / total
        p_y = y / total
        return -sum(p * np.log2(p) for p in [p_x, p_y] if p > 0)  # Sadece p > 0 için hesaplama
    return 0  # Eğer toplam sıfırsa entropi 0 olmalı
ilk_entropi=entropi(yes,no)    


def ozellik_entropisi(data,ozellik,etiket):
    toplam=len(data)
    ozellikler=data[ozellik].unique()
    weight_entropi=0
    for value in ozellikler:
        deger=data[data[ozellik]==value]
        yes_count=sum(deger[etiket]=="Yes")
        no_count=sum(deger[etiket]=="No")
        deger_entropisi=entropi(yes_count,no_count)
        weight_entropi+=(len(deger)/toplam)*deger_entropisi
    return weight_entropi 

def information_gain(data,ozellik,etiket="Class (Play Tennis)"):
    gelecek_ent=ozellik_entropisi(data, ozellik, etiket)
    return ilk_entropi-gelecek_ent
print(information_gain(data,"Feature 1 (Weather)"))
print(ozellik_entropisi(data, "Feature 1 (Weather)", "Class (Play Tennis)"))