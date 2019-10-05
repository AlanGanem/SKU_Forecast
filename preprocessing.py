# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
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
import TimeSeriesUtils as TSU
#import data
price_sales_promotion = pd.read_csv(r'C:\Users\ganem\Desktop\Arquivos pessoais\Projetos\Kaggle\SKU_forecast\price_sales_promotion.csv')
event_calendar = pd.read_csv(r'C:\Users\ganem\Desktop\Arquivos pessoais\Projetos\Kaggle\SKU_forecast\event_calendar.csv')
historical_volume = pd.read_csv(r'C:\Users\ganem\Desktop\Arquivos pessoais\Projetos\Kaggle\SKU_forecast\historical_volume.csv')
industry_soda_sales = pd.read_csv(r'C:\Users\ganem\Desktop\Arquivos pessoais\Projetos\Kaggle\SKU_forecast\industry_soda_sales.csv')
industry_volume = pd.read_csv(r'C:\Users\ganem\Desktop\Arquivos pessoais\Projetos\Kaggle\SKU_forecast\industry_volume.csv')
weather = pd.read_csv(r'C:\Users\ganem\Desktop\Arquivos pessoais\Projetos\Kaggle\SKU_forecast\weather.csv')
demographics = pd.read_csv(r'C:\Users\ganem\Desktop\Arquivos pessoais\Projetos\Kaggle\SKU_forecast\demographics.csv') # does not contaion YearMonth
sku_recommendation = pd.read_csv(r'C:\Users\ganem\Desktop\Arquivos pessoais\Projetos\Kaggle\SKU_forecast\sku_recommendation.csv')
volume_forecast = pd.read_csv(r'C:\Users\ganem\Desktop\Arquivos pessoais\Projetos\Kaggle\SKU_forecast\volume_forecast.csv')

# change date format
historical_volume['YearMonth'] = historical_volume['YearMonth'].apply(lambda x: pd.to_datetime(str(x)[:-2]+'/'+str(x)[-2:]))
price_sales_promotion['YearMonth'] = price_sales_promotion['YearMonth'].apply(lambda x: pd.to_datetime(str(x)[:-2]+'/'+str(x)[-2:]))
event_calendar['YearMonth'] = event_calendar['YearMonth'].apply(lambda x: pd.to_datetime(str(x)[:-2]+'/'+str(x)[-2:]))
industry_soda_sales['YearMonth'] = industry_soda_sales['YearMonth'].apply(lambda x:pd.to_datetime(str(x)[:-2]+'/'+str(x)[-2:]))
industry_volume['YearMonth'] = industry_volume['YearMonth'].apply(lambda x:pd.to_datetime(str(x)[:-2]+'/'+str(x)[-2:]))
weather['YearMonth'] = weather['YearMonth'].apply(lambda x: pd.to_datetime(str(x)[:-2]+'/'+str(x)[-2:]))

date_dfs = [
        historical_volume,
        price_sales_promotion,
        event_calendar,
        industry_soda_sales,
        industry_volume,
        weather
        ]

#join tables
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

# feature egineering session
merged_data['Month'] = merged_data['YearMonth'].dt.month
merged_data['Quarter'] = merged_data['YearMonth'].dt.quarter
merged_data['Discount'] = merged_data['Promotions']/(merged_data['Price']+0.1)
merged_data['Local_Relative_Temp'] = merged_data.groupby('Agency').apply(lambda x: x['Avg_Max_Temp']/x['Avg_Max_Temp'].max()).reset_index(level = 'Agency',drop =True)
merged_data.set_index('YearMonth', inplace = True)
merged_data = TSU.n_of_week_days(merged_data)
weekday_columns = merged_data[1]
merged_data = merged_data[0]

# set date as index
