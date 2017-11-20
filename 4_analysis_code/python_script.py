import os
import pandas as pd
import numpy as np
import math
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy
from scipy import stats
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.metrics import mean_squared_error, make_scorer

import warnings


os.chdir('/Users/lihuanyu/Desktop/平时学习/Kaggle Project/House_Prices/1_raw_data')

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# df_summary = df_train.describe()
full = train.append(test)

y = full.iloc[0:1460,full.columns.get_loc("SalePrice")]
log_y = np.log(y)

sns.distplot(y)
sns.distplot(log_y);
# check normality, realize log_y is more like Normal dist.
scipy.stats.probplot(log_x3,dist='norm',plot=plt)

col_ind = []
col_name = []
p_value = []
r_sqr = []
corr_coef = []

for ct in range(len(train.columns)-1):
	x = train.iloc[:,ct]
	colName = x.name
	if x.dtype == 'object':
		x = x.factorize()[0]
	# results = sm.OLS(y,x).fit()
	results = np.corrcoef(y,x)[0,1]
	# p = float(results.pvalues)
	# r2 = float(results.rsquared)
	col_ind.append(ct)
	col_name.append(colName)
	# p_value.append(p)
	# r_sqr.append(r2)
	corr_coef.append(results)

df1 = {'column_ind':pd.Series(col_ind),'column_name':pd.Series(col_name),
'corr_coef':pd.Series(corr_coef)
# 'p_value':pd.Series(p_value)
# ,'r_squared':pd.Series(r_sqr)
}

df = pd.DataFrame(df1)

# # feature selection
# df[df['p_value']<0.05]
# # 74 columns
# df[df['r_squared']>=0.30]
# # 40 columns
# df[(df['r_squared']>=0.30) & (df['p_value']<0.05)]
# #40 columns
# df[df['r_squared']>=0.50]
# #26 columns
df[abs(df['corr_coef'])>=0.2]
col_selected = df[abs(df['corr_coef'])>=0.2]
# df[df['r_squared']>=0.50]
#Based on these 26 columns, test collinearity.

# col_final = col_selected.drop([0,6,54,61,72,74])
col_final = col_selected

#length = len(col_final)
#ct1 = 0
#x1_ind = []
#x1_name = []
#x2_ind = []
#x2_name = []
## p_value1 = []
## r_sqr1 = []
#corr_coef2 = []
#while ct1<length:
#	col_indx1 = col_final.iloc[ct1,0]
#	x1 = train.iloc[:,col_indx1]
#	name1 = x1.name
#	if x1.dtype == 'object':
#		x1 = x1.factorize()[0]
#	
#	ct2 = ct1 + 1
#	while ct2<length:
#		col_indx2 = col_final.iloc[ct2,0]
#		x2 = train.iloc[:,col_indx2]
#		name2 = x2.name
#		if x2.dtype == 'object':
#			x2 = x2.factorize()[0]
#		results = np.corrcoef(x1,x2)[0,1]
#		# p = float(results.pvalues)
#		# r2 = float(results.rsquared)
#		x1_name.append(name1)
#		x2_name.append(name2)
#		x1_ind.append(col_indx1)
#		x2_ind.append(col_indx2)
#		# p_value1.append(p)
#		# r_sqr1.append(r2)
#		corr_coef2.append(results)
#		ct2+=1
#	ct1+=1	
#df2 = {'x1_ind':pd.Series(x1_ind),'x1_name':pd.Series(x1_name),
#'x2_ind':pd.Series(x2_ind),'x2_name':pd.Series(x2_name),
#'corr_coef':pd.Series(corr_coef2)
## 'p_value':pd.Series(p_value1),'r_squared':pd.Series(r_sqr1)
#}
#df2 = pd.DataFrame(df2)
#df2[abs(df2['corr_coef'])>=0.3].to_csv('col_reduce.csv')

col_final1 = col_final.drop([17,61,19,49,38,46,43,20,29,62,54,40,27,56,37])

col_final2 = col_final[col_final['corr_coef']>=0.50]
col_final2 = col_final2.drop([62,54,43,49,20])

#    column_ind  column_name  corr_coef
#17          17  OverallQual   0.790982
#19          19    YearBuilt   0.522897
#38          38  TotalBsmtSF   0.613581
#46          46    GrLivArea   0.708624
#61          61   GarageCars   0.640409


#check dist of independent variables
x1 = full.iloc[0:1460,full.columns.get_loc("OverallQual")]
x2 = full.iloc[0:1460,full.columns.get_loc("YearBuilt")]
x3 = full.iloc[0:1460,full.columns.get_loc("TotalBsmtSF")]
x4 = full.iloc[0:1460,full.columns.get_loc("GrLivArea")]
x5 = full.iloc[0:1460,full.columns.get_loc("GarageCars")]

log_x3 = np.log(x3[x3!=0])
log_x3.append(x3[x3==0])

log_x4 = np.log(x4[x4!=0])
log_x4.append(x4[x4==0])

scipy.stats.probplot(x2,dist='norm',plot=plt)

sns.distplot(x1)
sns.distplot(log_x4)

# after check, decide to take log-transformation for x3 and x4
# need to factorize x2


L=[]
#M = ['Id']
for x in col_final2['column_name']:
	L.append(x)
#	M.append(x)

df_train = full[L]
#20 variables + 1 SalePrice

# for ct in range(len(df_train.columns)):                   
# 	x = df_train.iloc[:,ct]
# 	name = x.name                  
# 	if x.dtype == 'object':
# 	    print(name)

#
##factorize
#df_train['LotShape']=df_train['LotShape'].factorize()[0]
#df_train['BsmtExposure']=df_train['BsmtExposure'].factorize()[0]
#df_train['BsmtFinType1']=df_train['BsmtFinType1'].factorize()[0]
#df_train['CentralAir']=df_train['CentralAir'].factorize()[0]
#df_train['Electrical']=df_train['Electrical'].factorize()[0]
#df_train['FireplaceQu']=df_train['FireplaceQu'].factorize()[0]
#df_train['GarageFinish']=df_train['GarageFinish'].factorize()[0]
#df_train['PavedDrive']=df_train['PavedDrive'].factorize()[0]
##
#df_train = df_train.drop(['2ndFlrSF'],axis=1)

#need to do the transformation here:
#transform x3 to log
df_train['TotalBsmtSF'][df_train['TotalBsmtSF']!=0] = np.log(df_train['TotalBsmtSF'][df_train['TotalBsmtSF']!=0])
#transform x4 to log
df_train['GrLivArea'][df_train['GrLivArea']!=0] = np.log(df_train['GrLivArea'][df_train['GrLivArea']!=0])

df_train['YearBuilt'] = df_train['YearBuilt'].factorize()[0]


df_train_train = df_train.iloc[0:1460,:]

lr = LinearRegression()
lr.fit(df_train_train,log_y)


##glm_Possion = sm.GLM(log_y,df_train_train,family=sm.families.Gaussian())
#linear_results = sm.OLS(log_y,df_train_train)
##res = glm_Possion.fit()
#res = linear_results.fit()

# df_train1 = train[M]
# df_train1['LotShape']=df_train1['LotShape'].factorize()[0]
# df_train1['BsmtExposure']=df_train1['BsmtExposure'].factorize()[0]
# df_train1['CentralAir']=df_train1['CentralAir'].factorize()[0]
# df_train1['PavedDrive']=df_train1['PavedDrive'].factorize()[0]

# df_train1['GarageFinish']=df_train1['GarageFinish'].factorize()[0]
# df_train1['FireplaceQu']=df_train1['FireplaceQu'].factorize()[0]
# df_train1['Electrical']=df_train1['Electrical'].factorize()[0]
# df_train1['BsmtFinType1']=df_train1['BsmtFinType1'].factorize()[0]
# df_train1.to_csv('final_train.csv')
test_x = df_train.iloc[1460:,:]
test_pre = lr.predict(test_x)

pre_results = np.exp(res.predict(test_x))
pre_results.to_csv('final_result.csv')
# test_x['LotShape']=test_x['LotShape'].factorize()[0]
# test_x['BsmtExposure']=test_x['BsmtExposure'].factorize()[0]
# test_x['BsmtFinType1']=test_x['BsmtFinType1'].factorize()[0]
# test_x['CentralAir']=test_x['CentralAir'].factorize()[0]
# test_x['Electrical']=test_x['Electrical'].factorize()[0]
# test_x['FireplaceQu']=test_x['FireplaceQu'].factorize()[0]
# test_x['GarageFinish']=test_x['GarageFinish'].factorize()[0]
# test_x['PavedDrive']=test_x['PavedDrive'].factorize()[0]
# test_x.to_csv('final_test.csv')


#target: minimize predict prices and real prices



