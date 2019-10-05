# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 19:03:58 2019

@author: ganem
"""
#imports
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

price_sales_promotion = pd.read_csv(r'C:\Users\ganem\Desktop\Arquivos pessoais\Projetos\Kaggle\SKU_forecast\price_sales_promotion.csv')
event_calendar = pd.read_csv(r'C:\Users\ganem\Desktop\Arquivos pessoais\Projetos\Kaggle\SKU_forecast\event_calendar.csv')
historical_volume = pd.read_csv(r'C:\Users\ganem\Desktop\Arquivos pessoais\Projetos\Kaggle\SKU_forecast\historical_volume.csv')
industry_soda_sales = pd.read_csv(r'C:\Users\ganem\Desktop\Arquivos pessoais\Projetos\Kaggle\SKU_forecast\industry_soda_sales.csv')
industry_volume = pd.read_csv(r'C:\Users\ganem\Desktop\Arquivos pessoais\Projetos\Kaggle\SKU_forecast\industry_volume.csv')
weather = pd.read_csv(r'C:\Users\ganem\Desktop\Arquivos pessoais\Projetos\Kaggle\SKU_forecast\weather.csv')
demographics = pd.read_csv(r'C:\Users\ganem\Desktop\Arquivos pessoais\Projetos\Kaggle\SKU_forecast\demographics.csv') # does not contaion YearMonth
sku_recommendation = pd.read_csv(r'C:\Users\ganem\Desktop\Arquivos pessoais\Projetos\Kaggle\SKU_forecast\sku_recommendation.csv')
volume_forecast = pd.read_csv(r'C:\Users\ganem\Desktop\Arquivos pessoais\Projetos\Kaggle\SKU_forecast\volume_forecast.csv')


historical_volume['YearMonth'] = historical_volume['YearMonth'].apply(lambda x: pd.to_datetime(str(x)[:-2]+'/'+str(x)[-2:]))
price_sales_promotion['YearMonth'] = price_sales_promotion['YearMonth'].apply(lambda x: pd.to_datetime(str(x)[:-2]+'/'+str(x)[-2:]))
event_calendar['YearMonth'] = event_calendar['YearMonth'].apply(lambda x: pd.to_datetime(str(x)[:-2]+'/'+str(x)[-2:]))
industry_soda_sales['YearMonth'] = industry_soda_sales['YearMonth'].apply(lambda x:pd.to_datetime(str(x)[:-2]+'/'+str(x)[-2:]))
industry_volume['YearMonth'] = industry_volume['YearMonth'].apply(lambda x:pd.to_datetime(str(x)[:-2]+'/'+str(x)[-2:]))
weather['YearMonth'] = weather['YearMonth'].apply(lambda x: pd.to_datetime(str(x)[:-2]+'/'+str(x)[-2:]))

sin1 = np.sin(np.linspace(0,10*np.pi,200))
sin2 = np.sin(5*np.linspace(0,10*np.pi,200))
sin3 = np.sin(10*np.linspace(0,10*np.pi,200))

plt.plot(sin1+sin2+sin3)


autocorr_arr = []
for lag in range(60):    
    autocorr_arr.append(weather[weather['Agency'] == 'Agency_01'].set_index('YearMonth')['Avg_Max_Temp'].diff().autocorr(-lag))
    #autocorr_arr.append(industry_volume['Industry_Volume'].autocorr(lag))
periodogram = signal.periodogram(weather[weather['Agency'] == 'Agency_01'].set_index('YearMonth')['Avg_Max_Temp'].diff().values[1:-2], fs = 1,scaling = 'spectrum',return_onesided = True)
plt.clf()
plt.plot(periodogram[0],periodogram[1])
plt.plot(autocorr_arr)
plt.plot(np.absolute(np.fft.rfft(np.array(autocorr_arr[:-2]))))
plt.plot(np.abs(np.fft.fft(sin1+sin2+sin3, n = 20)))
plt.plot(np.abs(np.fft.fft(sin2, n = 10)))


date_dfs = [
        historical_volume,
        price_sales_promotion,
        event_calendar,
        industry_soda_sales,
        industry_volume,
        weather
        ]

i = 0
for df_index in range(len(date_dfs)-1):
    if i == 0:
        df1 = date_dfs[df_index]
        df2 = date_dfs[df_index+1]
        df3 = pd.merge(df1,df2,right_on = list(set(df1.columns)&set(df2.columns)), left_on=list(set(df1.columns)&set(df2.columns)))
        print(df3.columns)
    else:
        df1 = df3
        df2 = date_dfs[df_index+1]
        df3 = pd.merge(df1,df2,right_on = list(set(df1.columns)&set(df2.columns)), left_on=list(set(df1.columns)&set(df2.columns)))
        print(df3.columns)
    i+=1
df1 = demographics
merged_data = pd.merge(df1,df3,right_on = list(set(df1.columns)&set(df3.columns)), left_on=list(set(df1.columns)&set(df3.columns)))

merged_data.set_index('YearMonth', inplace = True)

categorical_covariates = merged_data[['Agency','SKU']+list(event_calendar.columns)[1:]]
dependent_variable = merged_data[['Agency','SKU','Volume']]
continuous_covariates = merged_data[['Agency','SKU','Avg_Max_Temp','Industry_Volume','Soda_Volume','Avg_Population_2017', 'Avg_Yearly_Household_Income_2017']]

