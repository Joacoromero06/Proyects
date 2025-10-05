import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
file_path='temp.csv'
data=pd.read_csv(file_path,delimiter=';')
print(data.head)
print(data.columns)
for colum in ['promedio','máx','mín']:
    data[colum]=data[colum].str.replace(',','.').astype(float)
data['date']=pd.to_date(data['Fecha / Hora'])
"""
plt.figure()
plt.plot(daadate,data.promedio)
plt.show
"""