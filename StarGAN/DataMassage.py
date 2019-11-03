# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 19:58:54 2019

@author: asus
"""
from PIL import Image
import os
import pandas as pd

photo_dir = "D:\\UCB\\Dropbox\\Deep Learning\\RAF db\\"
out_dir = "C:\\Users\\asus\\StarGAN\\data\\RaFD\\"


file_name = pd.read_csv(photo_dir + "list_patition_label.txt",delim_whitespace=True,header = None)
file_name.columns = ["Name","label"]
file_name["Category"] = file_name.Name.apply(lambda x:x[0:5])
file_name.loc[file_name["Category"] == "test_","Category"] = "test" 
file_name["NewName"] = file_name.Name.apply(lambda x:x[:-4] + "_aligned.jpg")
file_name.loc[file_name["label"] == 1,"emo"] = "surprise"
file_name.loc[file_name["label"] == 2,"emo"] = "fear"
file_name.loc[file_name["label"] == 3,"emo"] = "disgust"
file_name.loc[file_name["label"] == 4,"emo"] = "happiness"
file_name.loc[file_name["label"] == 5,"emo"] = "sadness"
file_name.loc[file_name["label"] == 6,"emo"] = "anger"
file_name.loc[file_name["label"] == 7,"emo"] = "neutral"

for i in range(file_name.shape[0]):
    jpgfile = Image.open(photo_dir + "aligned\\" + file_name.NewName[i])
    jpgfile.save(out_dir + file_name.Category[i] +"\\"+ file_name.emo[i]+"\\" + file_name.Name[i] + ".JPEG", 'JPEG')