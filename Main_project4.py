
import tensorflow as tf
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, normalize, minmax_scale
from tensorflow.contrib import rnn
from tensorflow.contrib.layers import batch_norm as batch_norm
from pandas import ExcelWriter
import xlsxwriter
import numpy as np

from Classification_Function import Classification
path=r"C:\Users\mohammadrasoul.t\Desktop\temp\3G Site level_cut.xlsx"
data=pd.read_excel(path)
#data.head()
count=0
layer='SITENAME'

table = pd.pivot_table(data, values=data.columns,index=layer, columns="PERIOD_START_TIME")
total=pd.DataFrame()

for i in next(iter(table.columns.levels[0:1])):
    data_1=table[i]
    
    out=Classification(data_1)
    count=count+1
#Normalized_X=scale(data_1, axis=1)
#    Mi=Normalized_X.min()   
#    Normalized_X=(-1*Mi)+Normalized_X
#    with tf.Session(config=config) as sess:
#    
#        saver.restore(sess,"./TF_ModelwBN_for_Paper5.ckpt")
#        out=sess.run(prediction, feed_dict={X: Normalized_X})
##    with tf.Session(config=config) as sess:
##        out=sess.run(prediction, feed_dict={X: Normalized_X})
    out=pd.DataFrame(out)
    out.columns=['N','DS','SI','GD','SD','GI']
    out=out.idxmax(axis=1)
    out=pd.DataFrame(out)
    out.index=table.index
    if (count==1) :
        total=out
    total[str(i)]=out
    print (total)
    
with pd.ExcelWriter('KPI_all2.xlsx') as writer:
    total.to_excel(writer)
