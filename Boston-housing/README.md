![image](https://www.livetradingnews.com/wp-content/uploads/2017/01/home-sales-701x526.jpg)
<div style="text-align: center" > A Statistical Analysis & Machine Learning Workflow of House-Pricing </div>

<div style="text-align: center">This kernel is going to solve <font color="red"><b>House Pricing with Advanced Regression Analysis</b></font>, a popular machine learning dataset for <b>beginners</b>. I am going to share how I work with a dataset step by step  <b>from data preparation and data analysis to statistical tests and implementing machine learning models.</b> I will also describe the model results along with many other tips. Let's get started.</div>

<div style="text-align:center"> If there are any recommendations/changes you would like to see in this notebook, please <b>leave a comment</b> at the end of this kernel. Any feedback/constructive criticism would be genuinely appreciated. If you like this notebook or find this notebook helpful, Please feel free to <font color="red"><b>UPVOTE</b></font> and/or leave a comment.
 
<div> <b>This notebook is always a work in progress. So, please stay tuned for more to come.</b></div>

# Goals
This kernel hopes to accomplish many goals, to name a few...
* Learn/review/explain complex data science topics through write-ups. 
* Do a comprehensive data analysis along with visualizations. 
* Create models that are well equipped to predict better sale price of the houses. 

# Introduction
This kernel is the "regression siblings" of my other [ Classification kernel](https://www.kaggle.com/masumrumi/a-statistical-analysis-ml-workflow-of-titanic). As the name suggests, this kernel goes on a detailed analysis journey of most of the regression algorithms.  In addition to that, this kernel uses many charts and images to make things easier for readers to understand.
# 1: Importing Necessary Libraries and datasets


```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import matplotlib.gridspec as gridspec
from datetime import datetime
from scipy.stats import skew  # for some statistics
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.linear_model as linear_model
import matplotlib.style as style
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import missingno as msno

import os
print(os.listdir("../input"))

import warnings
warnings.filterwarnings('ignore')

# Any results you write to the current directory are saved as output.
```

    ['house-prices-advanced-regression-techniques']


# A Glimpse of the datasets.
> **Sample Train Dataset**


```python
## Import Trainning data. 
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>



> **Sample Test Dataset**


```python
## Import test data.
test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>...</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1461</td>
      <td>20</td>
      <td>RH</td>
      <td>80.0</td>
      <td>11622</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>120</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1462</td>
      <td>20</td>
      <td>RL</td>
      <td>81.0</td>
      <td>14267</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Gar2</td>
      <td>12500</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1463</td>
      <td>60</td>
      <td>RL</td>
      <td>74.0</td>
      <td>13830</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>MnPrv</td>
      <td>NaN</td>
      <td>0</td>
      <td>3</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1464</td>
      <td>60</td>
      <td>RL</td>
      <td>78.0</td>
      <td>9978</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>6</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1465</td>
      <td>120</td>
      <td>RL</td>
      <td>43.0</td>
      <td>5005</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>HLS</td>
      <td>AllPub</td>
      <td>...</td>
      <td>144</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>1</td>
      <td>2010</td>
      <td>WD</td>
      <td>Normal</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 80 columns</p>
</div>



# Describe the Datasets


```python
print (f"Train has {train.shape[0]} rows and {train.shape[1]} columns")
print (f"Test has {test.shape[0]} rows and {test.shape[1]} columns")
```

    Train has 1460 rows and 81 columns
    Test has 1459 rows and 80 columns


If you want to know more about why we are splitting dataset's into train and test, please check out this [kernel](https://www.kaggle.com/masumrumi/a-statistical-analysis-ml-workflow-of-titanic).


```python
# gives us statistical info about the numerical variables. 
train.describe().T
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>count</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Id</th>
      <td>1460.0</td>
      <td>730.500000</td>
      <td>421.610009</td>
      <td>1.0</td>
      <td>365.75</td>
      <td>730.5</td>
      <td>1095.25</td>
      <td>1460.0</td>
    </tr>
    <tr>
      <th>MSSubClass</th>
      <td>1460.0</td>
      <td>56.897260</td>
      <td>42.300571</td>
      <td>20.0</td>
      <td>20.00</td>
      <td>50.0</td>
      <td>70.00</td>
      <td>190.0</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>1201.0</td>
      <td>70.049958</td>
      <td>24.284752</td>
      <td>21.0</td>
      <td>59.00</td>
      <td>69.0</td>
      <td>80.00</td>
      <td>313.0</td>
    </tr>
    <tr>
      <th>LotArea</th>
      <td>1460.0</td>
      <td>10516.828082</td>
      <td>9981.264932</td>
      <td>1300.0</td>
      <td>7553.50</td>
      <td>9478.5</td>
      <td>11601.50</td>
      <td>215245.0</td>
    </tr>
    <tr>
      <th>OverallQual</th>
      <td>1460.0</td>
      <td>6.099315</td>
      <td>1.382997</td>
      <td>1.0</td>
      <td>5.00</td>
      <td>6.0</td>
      <td>7.00</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>OverallCond</th>
      <td>1460.0</td>
      <td>5.575342</td>
      <td>1.112799</td>
      <td>1.0</td>
      <td>5.00</td>
      <td>5.0</td>
      <td>6.00</td>
      <td>9.0</td>
    </tr>
    <tr>
      <th>YearBuilt</th>
      <td>1460.0</td>
      <td>1971.267808</td>
      <td>30.202904</td>
      <td>1872.0</td>
      <td>1954.00</td>
      <td>1973.0</td>
      <td>2000.00</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>YearRemodAdd</th>
      <td>1460.0</td>
      <td>1984.865753</td>
      <td>20.645407</td>
      <td>1950.0</td>
      <td>1967.00</td>
      <td>1994.0</td>
      <td>2004.00</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>1452.0</td>
      <td>103.685262</td>
      <td>181.066207</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>166.00</td>
      <td>1600.0</td>
    </tr>
    <tr>
      <th>BsmtFinSF1</th>
      <td>1460.0</td>
      <td>443.639726</td>
      <td>456.098091</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>383.5</td>
      <td>712.25</td>
      <td>5644.0</td>
    </tr>
    <tr>
      <th>BsmtFinSF2</th>
      <td>1460.0</td>
      <td>46.549315</td>
      <td>161.319273</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1474.0</td>
    </tr>
    <tr>
      <th>BsmtUnfSF</th>
      <td>1460.0</td>
      <td>567.240411</td>
      <td>441.866955</td>
      <td>0.0</td>
      <td>223.00</td>
      <td>477.5</td>
      <td>808.00</td>
      <td>2336.0</td>
    </tr>
    <tr>
      <th>TotalBsmtSF</th>
      <td>1460.0</td>
      <td>1057.429452</td>
      <td>438.705324</td>
      <td>0.0</td>
      <td>795.75</td>
      <td>991.5</td>
      <td>1298.25</td>
      <td>6110.0</td>
    </tr>
    <tr>
      <th>1stFlrSF</th>
      <td>1460.0</td>
      <td>1162.626712</td>
      <td>386.587738</td>
      <td>334.0</td>
      <td>882.00</td>
      <td>1087.0</td>
      <td>1391.25</td>
      <td>4692.0</td>
    </tr>
    <tr>
      <th>2ndFlrSF</th>
      <td>1460.0</td>
      <td>346.992466</td>
      <td>436.528436</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>728.00</td>
      <td>2065.0</td>
    </tr>
    <tr>
      <th>LowQualFinSF</th>
      <td>1460.0</td>
      <td>5.844521</td>
      <td>48.623081</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>572.0</td>
    </tr>
    <tr>
      <th>GrLivArea</th>
      <td>1460.0</td>
      <td>1515.463699</td>
      <td>525.480383</td>
      <td>334.0</td>
      <td>1129.50</td>
      <td>1464.0</td>
      <td>1776.75</td>
      <td>5642.0</td>
    </tr>
    <tr>
      <th>BsmtFullBath</th>
      <td>1460.0</td>
      <td>0.425342</td>
      <td>0.518911</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>BsmtHalfBath</th>
      <td>1460.0</td>
      <td>0.057534</td>
      <td>0.238753</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>FullBath</th>
      <td>1460.0</td>
      <td>1.565068</td>
      <td>0.550916</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
      <td>2.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>HalfBath</th>
      <td>1460.0</td>
      <td>0.382877</td>
      <td>0.502885</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>BedroomAbvGr</th>
      <td>1460.0</td>
      <td>2.866438</td>
      <td>0.815778</td>
      <td>0.0</td>
      <td>2.00</td>
      <td>3.0</td>
      <td>3.00</td>
      <td>8.0</td>
    </tr>
    <tr>
      <th>KitchenAbvGr</th>
      <td>1460.0</td>
      <td>1.046575</td>
      <td>0.220338</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>TotRmsAbvGrd</th>
      <td>1460.0</td>
      <td>6.517808</td>
      <td>1.625393</td>
      <td>2.0</td>
      <td>5.00</td>
      <td>6.0</td>
      <td>7.00</td>
      <td>14.0</td>
    </tr>
    <tr>
      <th>Fireplaces</th>
      <td>1460.0</td>
      <td>0.613014</td>
      <td>0.644666</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>3.0</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>1379.0</td>
      <td>1978.506164</td>
      <td>24.689725</td>
      <td>1900.0</td>
      <td>1961.00</td>
      <td>1980.0</td>
      <td>2002.00</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>GarageCars</th>
      <td>1460.0</td>
      <td>1.767123</td>
      <td>0.747315</td>
      <td>0.0</td>
      <td>1.00</td>
      <td>2.0</td>
      <td>2.00</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>GarageArea</th>
      <td>1460.0</td>
      <td>472.980137</td>
      <td>213.804841</td>
      <td>0.0</td>
      <td>334.50</td>
      <td>480.0</td>
      <td>576.00</td>
      <td>1418.0</td>
    </tr>
    <tr>
      <th>WoodDeckSF</th>
      <td>1460.0</td>
      <td>94.244521</td>
      <td>125.338794</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>168.00</td>
      <td>857.0</td>
    </tr>
    <tr>
      <th>OpenPorchSF</th>
      <td>1460.0</td>
      <td>46.660274</td>
      <td>66.256028</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>25.0</td>
      <td>68.00</td>
      <td>547.0</td>
    </tr>
    <tr>
      <th>EnclosedPorch</th>
      <td>1460.0</td>
      <td>21.954110</td>
      <td>61.119149</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>552.0</td>
    </tr>
    <tr>
      <th>3SsnPorch</th>
      <td>1460.0</td>
      <td>3.409589</td>
      <td>29.317331</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>508.0</td>
    </tr>
    <tr>
      <th>ScreenPorch</th>
      <td>1460.0</td>
      <td>15.060959</td>
      <td>55.757415</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>480.0</td>
    </tr>
    <tr>
      <th>PoolArea</th>
      <td>1460.0</td>
      <td>2.758904</td>
      <td>40.177307</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>738.0</td>
    </tr>
    <tr>
      <th>MiscVal</th>
      <td>1460.0</td>
      <td>43.489041</td>
      <td>496.123024</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>0.0</td>
      <td>0.00</td>
      <td>15500.0</td>
    </tr>
    <tr>
      <th>MoSold</th>
      <td>1460.0</td>
      <td>6.321918</td>
      <td>2.703626</td>
      <td>1.0</td>
      <td>5.00</td>
      <td>6.0</td>
      <td>8.00</td>
      <td>12.0</td>
    </tr>
    <tr>
      <th>YrSold</th>
      <td>1460.0</td>
      <td>2007.815753</td>
      <td>1.328095</td>
      <td>2006.0</td>
      <td>2007.00</td>
      <td>2008.0</td>
      <td>2009.00</td>
      <td>2010.0</td>
    </tr>
    <tr>
      <th>SalePrice</th>
      <td>1460.0</td>
      <td>180921.195890</td>
      <td>79442.502883</td>
      <td>34900.0</td>
      <td>129975.00</td>
      <td>163000.0</td>
      <td>214000.00</td>
      <td>755000.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Gives us information about the features. 
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 81 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             1460 non-null   int64  
     1   MSSubClass     1460 non-null   int64  
     2   MSZoning       1460 non-null   object 
     3   LotFrontage    1201 non-null   float64
     4   LotArea        1460 non-null   int64  
     5   Street         1460 non-null   object 
     6   Alley          91 non-null     object 
     7   LotShape       1460 non-null   object 
     8   LandContour    1460 non-null   object 
     9   Utilities      1460 non-null   object 
     10  LotConfig      1460 non-null   object 
     11  LandSlope      1460 non-null   object 
     12  Neighborhood   1460 non-null   object 
     13  Condition1     1460 non-null   object 
     14  Condition2     1460 non-null   object 
     15  BldgType       1460 non-null   object 
     16  HouseStyle     1460 non-null   object 
     17  OverallQual    1460 non-null   int64  
     18  OverallCond    1460 non-null   int64  
     19  YearBuilt      1460 non-null   int64  
     20  YearRemodAdd   1460 non-null   int64  
     21  RoofStyle      1460 non-null   object 
     22  RoofMatl       1460 non-null   object 
     23  Exterior1st    1460 non-null   object 
     24  Exterior2nd    1460 non-null   object 
     25  MasVnrType     1452 non-null   object 
     26  MasVnrArea     1452 non-null   float64
     27  ExterQual      1460 non-null   object 
     28  ExterCond      1460 non-null   object 
     29  Foundation     1460 non-null   object 
     30  BsmtQual       1423 non-null   object 
     31  BsmtCond       1423 non-null   object 
     32  BsmtExposure   1422 non-null   object 
     33  BsmtFinType1   1423 non-null   object 
     34  BsmtFinSF1     1460 non-null   int64  
     35  BsmtFinType2   1422 non-null   object 
     36  BsmtFinSF2     1460 non-null   int64  
     37  BsmtUnfSF      1460 non-null   int64  
     38  TotalBsmtSF    1460 non-null   int64  
     39  Heating        1460 non-null   object 
     40  HeatingQC      1460 non-null   object 
     41  CentralAir     1460 non-null   object 
     42  Electrical     1459 non-null   object 
     43  1stFlrSF       1460 non-null   int64  
     44  2ndFlrSF       1460 non-null   int64  
     45  LowQualFinSF   1460 non-null   int64  
     46  GrLivArea      1460 non-null   int64  
     47  BsmtFullBath   1460 non-null   int64  
     48  BsmtHalfBath   1460 non-null   int64  
     49  FullBath       1460 non-null   int64  
     50  HalfBath       1460 non-null   int64  
     51  BedroomAbvGr   1460 non-null   int64  
     52  KitchenAbvGr   1460 non-null   int64  
     53  KitchenQual    1460 non-null   object 
     54  TotRmsAbvGrd   1460 non-null   int64  
     55  Functional     1460 non-null   object 
     56  Fireplaces     1460 non-null   int64  
     57  FireplaceQu    770 non-null    object 
     58  GarageType     1379 non-null   object 
     59  GarageYrBlt    1379 non-null   float64
     60  GarageFinish   1379 non-null   object 
     61  GarageCars     1460 non-null   int64  
     62  GarageArea     1460 non-null   int64  
     63  GarageQual     1379 non-null   object 
     64  GarageCond     1379 non-null   object 
     65  PavedDrive     1460 non-null   object 
     66  WoodDeckSF     1460 non-null   int64  
     67  OpenPorchSF    1460 non-null   int64  
     68  EnclosedPorch  1460 non-null   int64  
     69  3SsnPorch      1460 non-null   int64  
     70  ScreenPorch    1460 non-null   int64  
     71  PoolArea       1460 non-null   int64  
     72  PoolQC         7 non-null      object 
     73  Fence          281 non-null    object 
     74  MiscFeature    54 non-null     object 
     75  MiscVal        1460 non-null   int64  
     76  MoSold         1460 non-null   int64  
     77  YrSold         1460 non-null   int64  
     78  SaleType       1460 non-null   object 
     79  SaleCondition  1460 non-null   object 
     80  SalePrice      1460 non-null   int64  
    dtypes: float64(3), int64(35), object(43)
    memory usage: 924.0+ KB



```python
## Gives use the count of different types of objects.
# train.get_dtype_counts()
```

## Checking for Missing Values

### Missing Train values


```python
msno.matrix(train);
```


![png](a-detailed-regression-guide-with-house-pricing_files/a-detailed-regression-guide-with-house-pricing_15_0.png)



```python
def missing_percentage(df):
    """This function takes a DataFrame(df) as input and returns two columns, total missing values and total missing values percentage"""
    ## the two following line may seem complicated but its actually very simple. 
    total = df.isnull().sum().sort_values(ascending = False)[df.isnull().sum().sort_values(ascending = False) != 0]
    percent = round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2)[round(df.isnull().sum().sort_values(ascending = False)/len(df)*100,2) != 0]
    return pd.concat([total, percent], axis=1, keys=['Total','Percent'])

missing_percentage(train)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PoolQC</th>
      <td>1453</td>
      <td>99.52</td>
    </tr>
    <tr>
      <th>MiscFeature</th>
      <td>1406</td>
      <td>96.30</td>
    </tr>
    <tr>
      <th>Alley</th>
      <td>1369</td>
      <td>93.77</td>
    </tr>
    <tr>
      <th>Fence</th>
      <td>1179</td>
      <td>80.75</td>
    </tr>
    <tr>
      <th>FireplaceQu</th>
      <td>690</td>
      <td>47.26</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>259</td>
      <td>17.74</td>
    </tr>
    <tr>
      <th>GarageCond</th>
      <td>81</td>
      <td>5.55</td>
    </tr>
    <tr>
      <th>GarageType</th>
      <td>81</td>
      <td>5.55</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>81</td>
      <td>5.55</td>
    </tr>
    <tr>
      <th>GarageFinish</th>
      <td>81</td>
      <td>5.55</td>
    </tr>
    <tr>
      <th>GarageQual</th>
      <td>81</td>
      <td>5.55</td>
    </tr>
    <tr>
      <th>BsmtExposure</th>
      <td>38</td>
      <td>2.60</td>
    </tr>
    <tr>
      <th>BsmtFinType2</th>
      <td>38</td>
      <td>2.60</td>
    </tr>
    <tr>
      <th>BsmtFinType1</th>
      <td>37</td>
      <td>2.53</td>
    </tr>
    <tr>
      <th>BsmtCond</th>
      <td>37</td>
      <td>2.53</td>
    </tr>
    <tr>
      <th>BsmtQual</th>
      <td>37</td>
      <td>2.53</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>8</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>MasVnrType</th>
      <td>8</td>
      <td>0.55</td>
    </tr>
    <tr>
      <th>Electrical</th>
      <td>1</td>
      <td>0.07</td>
    </tr>
  </tbody>
</table>
</div>



### Missing Train values


```python
msno.matrix(test);
```


![png](a-detailed-regression-guide-with-house-pricing_files/a-detailed-regression-guide-with-house-pricing_18_0.png)



```python
missing_percentage(test)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PoolQC</th>
      <td>1456</td>
      <td>99.79</td>
    </tr>
    <tr>
      <th>MiscFeature</th>
      <td>1408</td>
      <td>96.50</td>
    </tr>
    <tr>
      <th>Alley</th>
      <td>1352</td>
      <td>92.67</td>
    </tr>
    <tr>
      <th>Fence</th>
      <td>1169</td>
      <td>80.12</td>
    </tr>
    <tr>
      <th>FireplaceQu</th>
      <td>730</td>
      <td>50.03</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>227</td>
      <td>15.56</td>
    </tr>
    <tr>
      <th>GarageCond</th>
      <td>78</td>
      <td>5.35</td>
    </tr>
    <tr>
      <th>GarageQual</th>
      <td>78</td>
      <td>5.35</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>78</td>
      <td>5.35</td>
    </tr>
    <tr>
      <th>GarageFinish</th>
      <td>78</td>
      <td>5.35</td>
    </tr>
    <tr>
      <th>GarageType</th>
      <td>76</td>
      <td>5.21</td>
    </tr>
    <tr>
      <th>BsmtCond</th>
      <td>45</td>
      <td>3.08</td>
    </tr>
    <tr>
      <th>BsmtQual</th>
      <td>44</td>
      <td>3.02</td>
    </tr>
    <tr>
      <th>BsmtExposure</th>
      <td>44</td>
      <td>3.02</td>
    </tr>
    <tr>
      <th>BsmtFinType1</th>
      <td>42</td>
      <td>2.88</td>
    </tr>
    <tr>
      <th>BsmtFinType2</th>
      <td>42</td>
      <td>2.88</td>
    </tr>
    <tr>
      <th>MasVnrType</th>
      <td>16</td>
      <td>1.10</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>15</td>
      <td>1.03</td>
    </tr>
    <tr>
      <th>MSZoning</th>
      <td>4</td>
      <td>0.27</td>
    </tr>
    <tr>
      <th>BsmtHalfBath</th>
      <td>2</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>Utilities</th>
      <td>2</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>Functional</th>
      <td>2</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>BsmtFullBath</th>
      <td>2</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>BsmtFinSF2</th>
      <td>1</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>BsmtFinSF1</th>
      <td>1</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>Exterior2nd</th>
      <td>1</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>BsmtUnfSF</th>
      <td>1</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>TotalBsmtSF</th>
      <td>1</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>SaleType</th>
      <td>1</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>Exterior1st</th>
      <td>1</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>KitchenQual</th>
      <td>1</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>GarageArea</th>
      <td>1</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>GarageCars</th>
      <td>1</td>
      <td>0.07</td>
    </tr>
  </tbody>
</table>
</div>



# Observation
* There are multiple types of features. 
* Some features have missing values. 
* Most of the features are object( includes string values in the variable).

I want to focus on the target variable which is **SalePrice.** Let's create a histogram to see if the target variable is Normally distributed. If we want to create any linear model, it is essential that the features are normally distributed. This is one of the assumptions of multiple linear regression. I will explain more on this later.


```python
def plotting_3_chart(df, feature):
    ## Importing seaborn, matplotlab and scipy modules. 
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy import stats
    import matplotlib.style as style
    style.use('fivethirtyeight')

    ## Creating a customized chart. and giving in figsize and everything. 
    fig = plt.figure(constrained_layout=True, figsize=(12,8))
    ## creating a grid of 3 cols and 3 rows. 
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)
    #gs = fig3.add_gridspec(3, 3)

    ## Customizing the histogram grid. 
    ax1 = fig.add_subplot(grid[0, :2])
    ## Set the title. 
    ax1.set_title('Histogram')
    ## plot the histogram. 
    sns.distplot(df.loc[:,feature], norm_hist=True, ax = ax1)

    # customizing the QQ_plot. 
    ax2 = fig.add_subplot(grid[1, :2])
    ## Set the title. 
    ax2.set_title('QQ_plot')
    ## Plotting the QQ_Plot. 
    stats.probplot(df.loc[:,feature], plot = ax2)

    ## Customizing the Box Plot. 
    ax3 = fig.add_subplot(grid[:, 2])
    ## Set title. 
    ax3.set_title('Box Plot')
    ## Plotting the box plot. 
    sns.boxplot(df.loc[:,feature], orient='v', ax = ax3 );
    
plotting_3_chart(train, 'SalePrice')
```


![png](a-detailed-regression-guide-with-house-pricing_files/a-detailed-regression-guide-with-house-pricing_21_0.png)


These **three** charts above can tell us a lot about our target variable.
* Our target variable, **SalePrice** is not normally distributed.
* Our target variable is right-skewed. 
* There are multiple outliers in the variable.


**P.S.** 
* If you want to find out more about how to customize charts, try [this](https://matplotlib.org/tutorials/intermediate/gridspec.html#sphx-glr-tutorials-intermediate-gridspec-py) link. 
* If you are learning about Q-Q-plots for the first time. checkout [this](https://www.youtube.com/watch?v=smJBsZ4YQZw) video. 
* You can also check out [this](https://www.youtube.com/watch?v=9IcaQwQkE9I) one if you have some extra time. 

Let's find out how the sales price is distributed.


```python
#skewness and kurtosis
print("Skewness: " + str(train['SalePrice'].skew()))
print("Kurtosis: " + str(train['SalePrice'].kurt()))
```

    Skewness: 1.8828757597682129
    Kurtosis: 6.536281860064529


It looks like there are quite a bit Skewness and Kurtosis in the target variable. Let's talk about those a bit. 

<b>Skewness</b> 
* is the degree of distortion from the symmetrical bell curve or the normal curve. 
* So, a symmetrical distribution will have a skewness of "0". 
* There are two types of Skewness: <b>Positive and Negative.</b> 
* <b>Positive Skewness</b>(similar to our target variable distribution) means the tail on the right side of the distribution is longer and fatter. 
* In <b>positive Skewness </b> the mean and median will be greater than the mode similar to this dataset. Which means more houses were sold by less than the average price. 
* <b>Negative Skewness</b> means the tail on the left side of the distribution is longer and fatter.
* In <b>negative Skewness </b> the mean and median will be less than the mode. 
* Skewness differentiates in extreme values in one versus the other tail. 

Here is a picture to make more sense.  
![image](https://cdn-images-1.medium.com/max/1600/1*nj-Ch3AUFmkd0JUSOW_bTQ.jpeg)


<b>Kurtosis</b>
According to Wikipedia, 

*In probability theory and statistics, **Kurtosis** is the measure of the "tailedness" of the probability. distribution of a real-valued random variable.* So, In other words, **it is the measure of the extreme values(outliers) present in the distribution.** 

* There are three types of Kurtosis: <b>Mesokurtic, Leptokurtic, and Platykurtic</b>. 
* Mesokurtic is similar to the normal curve with the standard value of 3. This means that the extreme values of this distribution are similar to that of a normal distribution. 
* Leptokurtic Example of leptokurtic distributions are the T-distributions with small degrees of freedom.
* Platykurtic: Platykurtic describes a particular statistical distribution with thinner tails than a normal distribution. Because this distribution has thin tails, it has fewer outliers (e.g., extreme values three or more standard deviations from the mean) than do mesokurtic and leptokurtic distributions. 

![image](https://i2.wp.com/mvpprograms.com/help/images/KurtosisPict.jpg?resize=375%2C234)


You can read more about this from [this](https://codeburst.io/2-important-statistics-terms-you-need-to-know-in-data-science-skewness-and-kurtosis-388fef94eeaa) article. 

We can fix this by using different types of transformation(more on this later). However, before doing that, I want to find out the relationships among the target variable and other predictor variables. Let's find out.


```python
## Getting the correlation of all the features with target variable. 
(train.corr()**2)["SalePrice"].sort_values(ascending = False)[1:]
```




    OverallQual      0.625652
    GrLivArea        0.502149
    GarageCars       0.410124
    GarageArea       0.388667
    TotalBsmtSF      0.376481
    1stFlrSF         0.367057
    FullBath         0.314344
    TotRmsAbvGrd     0.284860
    YearBuilt        0.273422
    YearRemodAdd     0.257151
    GarageYrBlt      0.236548
    MasVnrArea       0.228000
    Fireplaces       0.218023
    BsmtFinSF1       0.149320
    LotFrontage      0.123763
    WoodDeckSF       0.105244
    2ndFlrSF         0.101974
    OpenPorchSF      0.099765
    HalfBath         0.080717
    LotArea          0.069613
    BsmtFullBath     0.051585
    BsmtUnfSF        0.046001
    BedroomAbvGr     0.028296
    KitchenAbvGr     0.018471
    EnclosedPorch    0.016532
    ScreenPorch      0.012420
    PoolArea         0.008538
    MSSubClass       0.007104
    OverallCond      0.006062
    MoSold           0.002156
    3SsnPorch        0.001988
    YrSold           0.000837
    LowQualFinSF     0.000656
    Id               0.000480
    MiscVal          0.000449
    BsmtHalfBath     0.000284
    BsmtFinSF2       0.000129
    Name: SalePrice, dtype: float64



These are the predictor variables sorted in a descending order starting with the most correlated one **OverallQual**. Let's put this one in a scatter plot and see how it looks.

### SalePrice vs OverallQual


```python
def customized_scatterplot(y, x):
        ## Sizing the plot. 
    style.use('fivethirtyeight')
    plt.subplots(figsize = (12,8))
    ## Plotting target variable with predictor variable(OverallQual)
    sns.scatterplot(y = y, x = x);
```


```python
customized_scatterplot(train.SalePrice, train.OverallQual)
```


![png](a-detailed-regression-guide-with-house-pricing_files/a-detailed-regression-guide-with-house-pricing_29_0.png)


**OverallQual** is a categorical variable, and a scatter plot is not the best way to visualize categorical variables. However, there is an apparent relationship between the two features. The price of the houses increases with the overall quality. Let's check out some more features to determine the outliers. Let's focus on the numerical variables this time.

### SalePrice vs GrLivArea


```python
customized_scatterplot(train.SalePrice, train.GrLivArea)
```


![png](a-detailed-regression-guide-with-house-pricing_files/a-detailed-regression-guide-with-house-pricing_32_0.png)


As you can see, there are two outliers in the plot above. We will get rid off them later. Let's look at another scatter plot with a different feature.

### SalePrice vs GarageArea


```python
customized_scatterplot(train.SalePrice, train.GarageArea);
```


![png](a-detailed-regression-guide-with-house-pricing_files/a-detailed-regression-guide-with-house-pricing_34_0.png)


And the next one..?
### SalePrice vs TotalBsmtSF


```python
customized_scatterplot(train.SalePrice, train.TotalBsmtSF)
```


![png](a-detailed-regression-guide-with-house-pricing_files/a-detailed-regression-guide-with-house-pricing_36_0.png)


and the next ?
### SalePrice vs 1stFlrSF


```python
customized_scatterplot(train.SalePrice, train['1stFlrSF']);
```


![png](a-detailed-regression-guide-with-house-pricing_files/a-detailed-regression-guide-with-house-pricing_38_0.png)


How about one more...


```python
customized_scatterplot(train.SalePrice, train.MasVnrArea);
```


![png](a-detailed-regression-guide-with-house-pricing_files/a-detailed-regression-guide-with-house-pricing_40_0.png)


Okay, I think we have seen enough. Let's discuss what we have found so far. 

# Observations
* Our target variable shows an unequal level of variance across most predictor(independent) variables. This is called **Heteroscedasticity(more explanation below)** and is a red flag for the multiple linear regression model.
* There are many outliers in the scatter plots above that took my attention. 

* The two on the top-right edge of **SalePrice vs. GrLivArea** seem to follow a trend, which can be explained by saying that "As the prices increased, so did the area. 
* However, The two on the bottom right of the same chart do not follow any trends. We will get rid of these two below.


```python

## Deleting those two values with outliers. 
train = train[train.GrLivArea < 4500]
train.reset_index(drop = True, inplace = True)

## save a copy of this dataset so that any changes later on can be compared side by side.
previous_train = train.copy()
```

As we look through these scatter plots, I realized that it is time to explain the assumptions of Multiple Linear Regression. Before building a multiple linear regression model, we need to check that these assumptions below are valid.
## Assumptions of Regression

* **Linearity ( Correct functional form )** 
* **Homoscedasticity ( Constant Error Variance )( vs Heteroscedasticity ). **
* **Independence of Errors ( vs Autocorrelation ) **
* **Multivariate Normality ( Normality of Errors ) **
* **No or little Multicollinearity. ** 

Since we fit a linear model, we assume that the relationship is **linear**, and the errors, or residuals, are pure random fluctuations around the true line. We expect that the variability in the response(dependent) variable doesn't increase as the value of the predictor(independent) increases, which is the assumptions of equal variance, also known as **Homoscedasticity**. We also assume that the observations are independent of one another(**No Multicollinearity**), and a correlation between sequential observations or auto-correlation is not there.

Now, these assumptions are prone to happen altogether. In other words, if we see one of these assumptions in the dataset, it's more likely that we may come across with others mentioned above. Therefore, we can find and fix various assumptions with a few unique techniques.

So, **How do we check regression assumptions? We fit a regression line and look for the variability of the response data along the regression line.** Let's apply this to each one of them.

**Linearity(Correct functional form):** 
Linear regression needs the relationship between each independent variable and the dependent variable to be linear. The linearity assumption can be tested with scatter plots. The following two examples depict two cases, where no or little linearity is present. 


```python
## Plot sizing. 
fig, (ax1, ax2) = plt.subplots(figsize = (12,8), ncols=2,sharey=False)
## Scatter plotting for SalePrice and GrLivArea. 
sns.scatterplot( x = train.GrLivArea, y = train.SalePrice,  ax=ax1)
## Putting a regression line. 
sns.regplot(x=train.GrLivArea, y=train.SalePrice, ax=ax1)

## Scatter plotting for SalePrice and MasVnrArea. 
sns.scatterplot(x = train.MasVnrArea,y = train.SalePrice, ax=ax2)
## regression line for MasVnrArea and SalePrice. 
sns.regplot(x=train.MasVnrArea, y=train.SalePrice, ax=ax2);
```


![png](a-detailed-regression-guide-with-house-pricing_files/a-detailed-regression-guide-with-house-pricing_44_0.png)


Here we are plotting our target variable with two independent variables **GrLivArea** and **MasVnrArea**. It's pretty apparent from the chart that there is a better linear relationship between **SalePrice** and **GrLivArea** than **SalePrice** and **MasVnrArea**. One thing to take note here, there are some outliers in the dataset. It is imperative to check for outliers since linear regression is sensitive to outlier effects. Sometimes we may be trying to fit a linear regression model when the data might not be so linear, or the function may need another degree of freedom to fit the data. In that case, we may need to change our function depending on the data to get the best possible fit. In addition to that, we can also check the residual plot, which tells us how is the error variance across the true line. Let's look at the residual plot for independent variable **GrLivArea** and our target variable **SalePrice **. 


```python
plt.subplots(figsize = (12,8))
sns.residplot(train.GrLivArea, train.SalePrice);
```


![png](a-detailed-regression-guide-with-house-pricing_files/a-detailed-regression-guide-with-house-pricing_46_0.png)


Ideally, if the assumptions are met, the residuals will be randomly scattered around the centerline of zero with no apparent pattern. The residual will look like an unstructured cloud of points centered around zero. However, our residual plot is anything but an unstructured cloud of points. Even though it seems like there is a linear relationship between the response variable and predictor variable, the residual plot looks more like a funnel. The error plot shows that as **GrLivArea** value increases, the variance also increases, which is the characteristics known as **Heteroscedasticity**. Let's break this down. 

**Homoscedasticity ( Constant Variance ):** 
The assumption of Homoscedasticity is crucial to linear regression models. Homoscedasticity describes a situation in which the error term or variance or the "noise" or random disturbance in the relationship between the independent variables and the dependent variable is the same across all values of the independent variable. In other words, there is a constant variance present in the response variable as the predictor variable increases. If the "noise" is not the same across the values of an independent variable like the residual plot above, we call that **Heteroscedasticity**. As you can tell, it is the opposite of **Homoscedasticity.**

<p><img src="https://www.dummies.com/wp-content/uploads/415147.image1.jpg" style="float:center"></img></p>

This plot above is an excellent example of Homoscedasticity. As you can see, the residual variance is the same as the value of the predictor variable increases. One way to fix this Heteroscedasticity is by using a transformation method like log-transformation or box-cox transformation. We will do that later.

**Multivariate Normality ( Normality of Errors):**
The linear regression analysis requires the dependent variable to be multivariate normally distributed. A histogram, box plot, or a Q-Q-Plot can check if the target variable is normally distributed. The goodness of fit test, e.g., the Kolmogorov-Smirnov test can check for normality in the dependent variable. We already know that our target variable does not follow a normal distribution. Let's bring back the three charts to show our target variable.


```python
plotting_3_chart(train, 'SalePrice')
```


![png](a-detailed-regression-guide-with-house-pricing_files/a-detailed-regression-guide-with-house-pricing_48_0.png)


Now, let's make sure that the target variable follows a normal distribution. If you want to learn more about the probability plot(Q-Q plot), try [this](https://www.youtube.com/watch?v=smJBsZ4YQZw) video. You can also check out [this](https://www.youtube.com/watch?v=9IcaQwQkE9I) one if you have some extra time.


```python
## trainsforming target variable using numpy.log1p, 
train["SalePrice"] = np.log1p(train["SalePrice"])

## Plotting the newly transformed response variable
plotting_3_chart(train, 'SalePrice')
```


![png](a-detailed-regression-guide-with-house-pricing_files/a-detailed-regression-guide-with-house-pricing_50_0.png)


As you can see, the log transformation removes the normality of errors, which solves most of the other errors we talked about above. Let's make a comparison of the pre-transformed and post-transformed state of residual plots. 


```python
## Customizing grid for two plots. 
fig, (ax1, ax2) = plt.subplots(figsize = (15,6), 
                               ncols=2, 
                               sharey = False, 
                               sharex=False
                              )
## doing the first scatter plot. 
sns.residplot(x = previous_train.GrLivArea, y = previous_train.SalePrice, ax = ax1)
## doing the scatter plot for GrLivArea and SalePrice. 
sns.residplot(x = train.GrLivArea, y = train.SalePrice, ax = ax2);
```


![png](a-detailed-regression-guide-with-house-pricing_files/a-detailed-regression-guide-with-house-pricing_52_0.png)


Here, we see that the pre-transformed chart on the left has heteroscedasticity, and the post-transformed chart on the right has Homoscedasticity(almost an equal amount of variance across the zero lines). It looks like a blob of data points and doesn't seem to give away any relationships. That's the sort of relationship we would like to see to avoid some of these assumptions. 

**No or Little multicollinearity:** 
Multicollinearity is when there is a strong correlation between independent variables. Linear regression or multilinear regression requires independent variables to have little or no similar features. Multicollinearity can lead to a variety of problems, including:
* The effect of predictor variables estimated by our regression will depend on what other variables are included in our model. 
* Predictors can have wildly different results depending on the observations in our sample, and small changes in samples can result in very different estimated effects. 
* With very high multicollinearity, the inverse matrix, the computer calculates may not be accurate. 
* We can no longer interpret a coefficient on a variable as the effect on the target of a one-unit increase in that variable holding the other variables constant. The reason behind that is, when predictors are strongly correlated, there is not a scenario in which one variable can change without a conditional change in another variable.

Heatmap is an excellent way to identify whether there is multicollinearity or not. The best way to solve multicollinearity is to use regularization methods like Ridge or Lasso.


```python
## Plot fig sizing. 
style.use('ggplot')
sns.set_style('whitegrid')
plt.subplots(figsize = (30,20))
## Plotting heatmap. 

# Generate a mask for the upper triangle (taken from seaborn example gallery)
mask = np.zeros_like(train.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True


sns.heatmap(train.corr(), 
            cmap=sns.diverging_palette(20, 220, n=200), 
            mask = mask, 
            annot=True, 
            center = 0, 
           );
## Give title. 
plt.title("Heatmap of all the Features", fontsize = 30);
```


![png](a-detailed-regression-guide-with-house-pricing_files/a-detailed-regression-guide-with-house-pricing_54_0.png)


## Observation. 
As we can see, the multicollinearity still exists in various features. However, we will keep them for now for the sake of learning and let the models(e.x. Regularization models such as Lasso, Ridge) do the clean up later on. Let's go through some of the correlations that still exists. 

* There is 0.83 or 83% correlation between **GarageYrBlt** and **YearBuilt**. 
* 83% correlation between **TotRmsAbvGrd ** and **GrLivArea**. 
* 89% correlation between **GarageCars** and **GarageArea**. 
* Similarly many other features such as**BsmtUnfSF**, **FullBath** have good correlation with other independent feature.

If I were using only multiple linear regression, I would be deleting these features from the dataset to fit better multiple linear regression algorithms. However, we will be using many algorithms as scikit learn modules makes it easy to implement them and get the best possible outcome. Therefore, we will keep all the features for now. 

<h3>Resources:</h3>
<ul>
    <li><a href="https://www.statisticssolutions.com/assumptions-of-linear-regression/">Assumptions of Linear Regression</a></li>
    <li><a href="https://www.statisticssolutions.com/assumptions-of-multiple-linear-regression/">Assumptions of Multiple Linear Regression</a></li>
    <li><a href="https://www.youtube.com/watch?v=0MFpOQRY0rw/"> Youtube: All regression assumptions explained!<a/></li>
</ul>

# Feature engineering


```python
## Dropping the "Id" from train and test set. 
# train.drop(columns=['Id'],axis=1, inplace=True)

train.drop(columns=['Id'],axis=1, inplace=True)
test.drop(columns=['Id'],axis=1, inplace=True)

## Saving the target values in "y_train". 
y = train['SalePrice'].reset_index(drop=True)



# getting a copy of train
previous_train = train.copy()
```


```python
# quantitative = [f for f in train.columns if train.dtypes[f] != 'object']
# qualitative = [f for f in train.columns if train.dtypes[f] == 'object']

# def encode(df, feature, target_feature):
#     """
#     This function takes a dataframe, a feature(a categorical feature) and a target_feature(the feature that should be used for encoding)
#     and returns a new feature with the original feature name + postfix(_E). 
#     This new feature consists of encoded value of unique original value but the values are weighted(incremented) based on the 
#     mean of target_feature and grouped by the feature itself.
#     """
#     ordering = pd.DataFrame()
#     ordering['val'] = df[feature].unique()
#     ordering.index = ordering.val
#     ordering['spmean'] = df[[feature, target_feature]].groupby(feature).mean()[target_feature]
#     ordering = ordering.sort_values('spmean')
#     ordering['ordering'] = range(1, ordering.shape[0]+1)
#     ordering = ordering['ordering'].to_dict()
    
#     for cat, o in ordering.items():
#         df.loc[df[feature] == cat, feature+'_E'] = o
    
# qual_encoded = []
# for q in qualitative:  
#     encode(train, q, 'SalePrice')
#     qual_encoded.append(q+'_E')
# print(qual_encoded)
```


```python
## Combining train and test datasets together so that we can do all the work at once. 
all_data = pd.concat((train, test)).reset_index(drop = True)
## Dropping the target variable. 
all_data.drop(['SalePrice'], axis = 1, inplace = True)
```

## Dealing with Missing Values
> **Missing data in train and test data(all_data)**


```python
missing_percentage(all_data)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>PoolQC</th>
      <td>2908</td>
      <td>99.69</td>
    </tr>
    <tr>
      <th>MiscFeature</th>
      <td>2812</td>
      <td>96.40</td>
    </tr>
    <tr>
      <th>Alley</th>
      <td>2719</td>
      <td>93.21</td>
    </tr>
    <tr>
      <th>Fence</th>
      <td>2346</td>
      <td>80.43</td>
    </tr>
    <tr>
      <th>FireplaceQu</th>
      <td>1420</td>
      <td>48.68</td>
    </tr>
    <tr>
      <th>LotFrontage</th>
      <td>486</td>
      <td>16.66</td>
    </tr>
    <tr>
      <th>GarageCond</th>
      <td>159</td>
      <td>5.45</td>
    </tr>
    <tr>
      <th>GarageQual</th>
      <td>159</td>
      <td>5.45</td>
    </tr>
    <tr>
      <th>GarageYrBlt</th>
      <td>159</td>
      <td>5.45</td>
    </tr>
    <tr>
      <th>GarageFinish</th>
      <td>159</td>
      <td>5.45</td>
    </tr>
    <tr>
      <th>GarageType</th>
      <td>157</td>
      <td>5.38</td>
    </tr>
    <tr>
      <th>BsmtCond</th>
      <td>82</td>
      <td>2.81</td>
    </tr>
    <tr>
      <th>BsmtExposure</th>
      <td>82</td>
      <td>2.81</td>
    </tr>
    <tr>
      <th>BsmtQual</th>
      <td>81</td>
      <td>2.78</td>
    </tr>
    <tr>
      <th>BsmtFinType2</th>
      <td>80</td>
      <td>2.74</td>
    </tr>
    <tr>
      <th>BsmtFinType1</th>
      <td>79</td>
      <td>2.71</td>
    </tr>
    <tr>
      <th>MasVnrType</th>
      <td>24</td>
      <td>0.82</td>
    </tr>
    <tr>
      <th>MasVnrArea</th>
      <td>23</td>
      <td>0.79</td>
    </tr>
    <tr>
      <th>MSZoning</th>
      <td>4</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>BsmtHalfBath</th>
      <td>2</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>Utilities</th>
      <td>2</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>Functional</th>
      <td>2</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>BsmtFullBath</th>
      <td>2</td>
      <td>0.07</td>
    </tr>
    <tr>
      <th>BsmtFinSF2</th>
      <td>1</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>BsmtFinSF1</th>
      <td>1</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>Exterior2nd</th>
      <td>1</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>BsmtUnfSF</th>
      <td>1</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>TotalBsmtSF</th>
      <td>1</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>Exterior1st</th>
      <td>1</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>SaleType</th>
      <td>1</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>Electrical</th>
      <td>1</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>KitchenQual</th>
      <td>1</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>GarageArea</th>
      <td>1</td>
      <td>0.03</td>
    </tr>
    <tr>
      <th>GarageCars</th>
      <td>1</td>
      <td>0.03</td>
    </tr>
  </tbody>
</table>
</div>



> **Imputing Missing Values**


```python
## Some missing values are intentionally left blank, for example: In the Alley feature 
## there are blank values meaning that there are no alley's in that specific house. 
missing_val_col = ["Alley", 
                   "PoolQC", 
                   "MiscFeature",
                   "Fence",
                   "FireplaceQu",
                   "GarageType",
                   "GarageFinish",
                   "GarageQual",
                   "GarageCond",
                   'BsmtQual',
                   'BsmtCond',
                   'BsmtExposure',
                   'BsmtFinType1',
                   'BsmtFinType2',
                   'MasVnrType']

for i in missing_val_col:
    all_data[i] = all_data[i].fillna('None')
```


```python
## In the following features the null values are there for a purpose, so we replace them with "0"
missing_val_col2 = ['BsmtFinSF1',
                    'BsmtFinSF2',
                    'BsmtUnfSF',
                    'TotalBsmtSF',
                    'BsmtFullBath', 
                    'BsmtHalfBath', 
                    'GarageYrBlt',
                    'GarageArea',
                    'GarageCars',
                    'MasVnrArea']

for i in missing_val_col2:
    all_data[i] = all_data[i].fillna(0)
    
## Replaced all missing values in LotFrontage by imputing the median value of each neighborhood. 
all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform( lambda x: x.fillna(x.mean()))
```


```python
## the "OverallCond" and "OverallQual" of the house. 
# all_data['OverallCond'] = all_data['OverallCond'].astype(str) 
# all_data['OverallQual'] = all_data['OverallQual'].astype(str)

## Zoning class are given in numerical; therefore converted to categorical variables. 
all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)
all_data['MSZoning'] = all_data.groupby('MSSubClass')['MSZoning'].transform(lambda x: x.fillna(x.mode()[0]))

## Important years and months that should be categorical variables not numerical. 
# all_data['YearBuilt'] = all_data['YearBuilt'].astype(str)
# all_data['YearRemodAdd'] = all_data['YearRemodAdd'].astype(str)
# all_data['GarageYrBlt'] = all_data['GarageYrBlt'].astype(str)
all_data['YrSold'] = all_data['YrSold'].astype(str)
all_data['MoSold'] = all_data['MoSold'].astype(str) 
```


```python
all_data['Functional'] = all_data['Functional'].fillna('Typ') 
all_data['Utilities'] = all_data['Utilities'].fillna('AllPub') 
all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0]) 
all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])
all_data['KitchenQual'] = all_data['KitchenQual'].fillna("TA") 
all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])
all_data['Electrical'] = all_data['Electrical'].fillna("SBrkr") 

```


```python
missing_percentage(all_data)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Total</th>
      <th>Percent</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



So, there are no missing value left. 

## Fixing Skewness


```python
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)

skewed_feats
```




    MiscVal          21.939672
    PoolArea         17.688664
    LotArea          13.109495
    LowQualFinSF     12.084539
    3SsnPorch        11.372080
    KitchenAbvGr      4.300550
    BsmtFinSF2        4.144503
    EnclosedPorch     4.002344
    ScreenPorch       3.945101
    BsmtHalfBath      3.929996
    MasVnrArea        2.621719
    OpenPorchSF       2.529358
    WoodDeckSF        1.844792
    1stFlrSF          1.257286
    GrLivArea         1.068750
    LotFrontage       1.058803
    BsmtFinSF1        0.980645
    BsmtUnfSF         0.919688
    2ndFlrSF          0.861556
    TotRmsAbvGrd      0.749232
    Fireplaces        0.725278
    HalfBath          0.696666
    TotalBsmtSF       0.671751
    BsmtFullBath      0.622415
    OverallCond       0.569314
    BedroomAbvGr      0.326568
    GarageArea        0.216857
    OverallQual       0.189591
    FullBath          0.165514
    GarageCars       -0.219297
    YearRemodAdd     -0.450134
    YearBuilt        -0.599194
    GarageYrBlt      -3.904632
    dtype: float64




```python
sns.distplot(all_data['1stFlrSF']);
```


![png](a-detailed-regression-guide-with-house-pricing_files/a-detailed-regression-guide-with-house-pricing_70_0.png)



```python
## Fixing Skewed features using boxcox transformation. 


def fixing_skewness(df):
    """
    This function takes in a dataframe and return fixed skewed dataframe
    """
    ## Import necessary modules 
    from scipy.stats import skew
    from scipy.special import boxcox1p
    from scipy.stats import boxcox_normmax
    
    ## Getting all the data that are not of "object" type. 
    numeric_feats = df.dtypes[df.dtypes != "object"].index

    # Check the skew of all numerical features
    skewed_feats = df[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
    high_skew = skewed_feats[abs(skewed_feats) > 0.5]
    skewed_features = high_skew.index

    for feat in skewed_features:
        df[feat] = boxcox1p(df[feat], boxcox_normmax(df[feat] + 1))

fixing_skewness(all_data)
```


```python
sns.distplot(all_data['1stFlrSF']);
```


![png](a-detailed-regression-guide-with-house-pricing_files/a-detailed-regression-guide-with-house-pricing_72_0.png)


## Creating New Features


```python
# feture engineering a new feature "TotalFS"
all_data['TotalSF'] = (all_data['TotalBsmtSF'] 
                       + all_data['1stFlrSF'] 
                       + all_data['2ndFlrSF'])

all_data['YrBltAndRemod'] = all_data['YearBuilt'] + all_data['YearRemodAdd']

all_data['Total_sqr_footage'] = (all_data['BsmtFinSF1'] 
                                 + all_data['BsmtFinSF2'] 
                                 + all_data['1stFlrSF'] 
                                 + all_data['2ndFlrSF']
                                )
                                 

all_data['Total_Bathrooms'] = (all_data['FullBath'] 
                               + (0.5 * all_data['HalfBath']) 
                               + all_data['BsmtFullBath'] 
                               + (0.5 * all_data['BsmtHalfBath'])
                              )
                               

all_data['Total_porch_sf'] = (all_data['OpenPorchSF'] 
                              + all_data['3SsnPorch'] 
                              + all_data['EnclosedPorch'] 
                              + all_data['ScreenPorch'] 
                              + all_data['WoodDeckSF']
                             )
                              
                              

```


```python
all_data['haspool'] = all_data['PoolArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['has2ndfloor'] = all_data['2ndFlrSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasgarage'] = all_data['GarageArea'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasbsmt'] = all_data['TotalBsmtSF'].apply(lambda x: 1 if x > 0 else 0)
all_data['hasfireplace'] = all_data['Fireplaces'].apply(lambda x: 1 if x > 0 else 0)
```


```python
all_data.shape
```




    (2917, 89)



## Deleting features


```python
all_data = all_data.drop(['Utilities', 'Street', 'PoolQC',], axis=1)
```

## Creating Dummy Variables. 



```python
## Creating dummy variable 
final_features = pd.get_dummies(all_data).reset_index(drop=True)
final_features.shape
```




    (2917, 333)




```python
X = final_features.iloc[:len(y), :]

X_sub = final_features.iloc[len(y):, :]
```


```python
outliers = [30, 88, 462, 631, 1322]
X = X.drop(X.index[outliers])
y = y.drop(y.index[outliers])
```


```python
counts = X.BsmtUnfSF.value_counts()
```


```python
counts.iloc[0]
```




    117




```python
for i in X.columns:
    counts = X[i].value_counts()
    print (counts)
```

    17.840337    142
    19.692955     70
    21.443061     69
    15.860626     56
    20.579604     52
                ... 
    30.245223      1
    26.381204      1
    30.655039      1
    30.518824      1
    28.852705      1
    Name: LotFrontage, Length: 132, dtype: int64
    13.480169    25
    14.117918    24
    13.084300    17
    13.973432    14
    14.383736    14
                 ..
    14.297174     1
    14.170671     1
    12.981584     1
    13.493508     1
    14.802122     1
    Name: LotArea, Length: 1068, dtype: int64
    5     396
    6     374
    7     318
    8     167
    4     115
    9      43
    3      19
    10     16
    2       3
    1       2
    Name: OverallQual, dtype: int64
    3.991517    816
    4.679501    252
    5.348041    205
    6.000033     72
    3.280100     56
    2.539440     25
    6.637670     22
    1.760360      4
    0.926401      1
    Name: OverallCond, dtype: int64
    2.803702e+51    66
    2.781514e+51    64
    2.759490e+51    54
    2.826056e+51    48
    2.737630e+51    45
                    ..
    1.315931e+51     1
    1.230930e+51     1
    9.558503e+50     1
    1.241267e+51     1
    1.160804e+51     1
    Name: YearBuilt, Length: 112, dtype: int64
    1950    177
    2006     97
    2007     75
    2005     73
    2004     62
           ... 
    1982      6
    1983      5
    1952      5
    1986      5
    1951      4
    Name: YearRemodAdd, Length: 61, dtype: int64
    0.000000     866
    18.672689      8
    12.010890      8
    14.642917      8
    5.433718       7
                ... 
    23.416413      1
    42.724168      1
    22.239998      1
    26.248403      1
    4.372150       1
    Name: MasVnrArea, Length: 323, dtype: int64
    0.000000      464
    12.329405      12
    9.052858        9
    137.662974      5
    130.778407      5
                 ... 
    100.018543      1
    237.061765      1
    176.804868      1
    51.345625       1
    217.555782      1
    Name: BsmtFinSF1, Length: 635, dtype: int64
    0.000000     1287
    8.279525        5
    10.125437       3
    11.951035       2
    9.492405        2
                 ... 
    11.574678       1
    7.441943        1
    4.749196        1
    8.185842        1
    7.160610        1
    Name: BsmtFinSF2, Length: 143, dtype: int64
    0.000000      117
    77.387613       9
    52.884732       8
    67.066133       7
    45.613017       7
                 ... 
    60.118950       1
    120.909168      1
    133.570538      1
    107.671755      1
    93.323274       1
    Name: BsmtUnfSF, Length: 776, dtype: int64
    0.000000      37
    425.950955    34
    341.648447    17
    446.644084    15
    501.188603    14
                  ..
    530.241399     1
    476.156877     1
    764.446500     1
    771.213184     1
    394.640480     1
    Name: TotalBsmtSF, Length: 718, dtype: int64
    5.946177    24
    6.088025    16
    5.987645    14
    5.931821    12
    5.972366    12
                ..
    6.319806     1
    5.597880     1
    6.750047     1
    6.250418     1
    5.560106     1
    Name: 1stFlrSF, Length: 748, dtype: int64
    0.000000       825
    869.944539      10
    595.353605       9
    801.007115       8
    646.587765       8
                  ... 
    1551.995403      1
    1416.676846      1
    515.120311       1
    1157.214536      1
    972.425816       1
    Name: 2ndFlrSF, Length: 415, dtype: int64
    0.000000    1428
    3.731031       3
    4.739339       2
    4.962539       1
    4.918806       1
    4.790020       1
    4.801244       1
    4.461413       1
    4.910965       1
    4.374295       1
    4.963741       1
    3.436951       1
    4.013260       1
    4.191825       1
    4.137690       1
    4.758431       1
    4.780227       1
    4.836677       1
    4.455786       1
    5.028404       1
    4.979144       1
    4.793249       1
    4.921396       1
    Name: LowQualFinSF, dtype: int64
    7.507207    21
    7.735276    14
    7.549095    11
    7.484287    10
    8.152571    10
                ..
    8.393937     1
    8.647890     1
    7.813492     1
    8.564072     1
    8.583852     1
    Name: GrLivArea, Length: 857, dtype: int64
    0.000000    851
    0.993440    587
    1.978051     14
    2.956972      1
    Name: BsmtFullBath, dtype: int64
    0.000000    1371
    0.710895      80
    1.143642       2
    Name: BsmtHalfBath, dtype: int64
    2    765
    1    647
    3     32
    0      9
    Name: FullBath, dtype: int64
    0.000000    908
    1.068837    533
    2.237197     12
    Name: HalfBath, dtype: int64
    3    799
    2    356
    4    213
    1     50
    5     21
    6      7
    0      6
    8      1
    Name: BedroomAbvGr, dtype: int64
    0.750957    1385
    1.248543      65
    1.630565       2
    0.000000       1
    Name: KitchenAbvGr, dtype: int64
    1.996577    399
    2.137369    328
    1.834659    274
    2.261968    187
    1.643995     97
    2.373753     75
    2.475142     47
    2.567925     17
    1.411883     17
    2.653465     10
    2.806843      1
    1.114642      1
    Name: TotRmsAbvGrd, dtype: int64
    0.000000    688
    0.903334    646
    1.688254    115
    2.404976      4
    Name: Fireplaces, dtype: int64
    0.000000e+00    80
    8.470611e+56    65
    8.545305e+56    59
    8.396532e+56    53
    8.323065e+56    50
                    ..
    4.449082e+56     1
    3.473051e+56     1
    3.537778e+56     1
    3.285507e+56     1
    4.212100e+56     1
    Name: GarageYrBlt, Length: 98, dtype: int64
    2.0    822
    1.0    367
    3.0    179
    0.0     80
    4.0      5
    Name: GarageCars, dtype: int64
    0.0      80
    440.0    49
    576.0    47
    240.0    38
    484.0    34
             ..
    616.0     1
    445.0     1
    872.0     1
    696.0     1
    789.0     1
    Name: GarageArea, Length: 440, dtype: int64
    0.000000     757
    42.245702     37
    27.547833     36
    35.011824     33
    31.064436     31
                ... 
    41.958652      1
    52.076157      1
    51.821307      1
    36.428669      1
    78.415692      1
    Name: WoodDeckSF, Length: 274, dtype: int64
    0.000000     653
    9.105438      29
    10.637927     22
    6.548761      21
    10.276091     19
                ... 
    24.241322      1
    23.384421      1
    16.079297      1
    13.643401      1
    25.490355      1
    Name: OpenPorchSF, Length: 200, dtype: int64
    0.000000     1248
    11.252030      15
    10.557010       6
    14.644807       5
    11.574808       5
                 ... 
    10.131463       1
    8.890646        1
    16.334336       1
    14.591443       1
    15.929118       1
    Name: EnclosedPorch, Length: 119, dtype: int64
    0.000000    1429
    6.174266       3
    5.955943       2
    6.272811       2
    6.535696       2
    7.038144       1
    6.968074       1
    6.122528       1
    5.812499       1
    5.916321       1
    5.393928       1
    6.719441       1
    6.395151       1
    6.677010       1
    7.817018       1
    7.114652       1
    6.288641       1
    7.477204       1
    3.560137       1
    6.041500       1
    Name: 3SsnPorch, dtype: int64
    0.000000     1337
    24.456268       6
    19.077822       5
    26.506769       5
    23.642560       4
                 ... 
    35.575499       1
    29.374551       1
    31.988628       1
    34.548018       1
    23.984879       1
    Name: ScreenPorch, Length: 76, dtype: int64
    0.000000    1447
    5.606854       1
    5.703226       1
    5.519096       1
    5.441078       1
    5.491330       1
    5.430885       1
    Name: PoolArea, dtype: int64
    0.000000     1401
    6.315042       11
    6.562433        8
    6.937474        5
    8.122680        4
    6.445490        4
    6.765357        4
    7.543164        2
    6.517095        2
    8.763610        1
    7.495121        1
    6.801927        1
    8.377489        1
    6.688485        1
    7.086969        1
    4.149193        1
    7.717485        1
    7.633620        1
    6.167518        1
    9.764988        1
    10.498741       1
    Name: MiscVal, dtype: int64
    431.897132     21
    424.953764     10
    444.872713     10
    411.017210      9
    452.631730      7
                   ..
    2308.290185     1
    471.090201      1
    744.115628      1
    919.389872      1
    759.641390      1
    Name: TotalSF, Length: 1247, dtype: int64
    2.803702e+51    66
    2.781514e+51    64
    2.759490e+51    54
    2.826056e+51    48
    2.737630e+51    45
                    ..
    1.315931e+51     1
    1.230930e+51     1
    9.558503e+50     1
    1.241267e+51     1
    1.160804e+51     1
    Name: YrBltAndRemod, Length: 112, dtype: int64
    6.088025       6
    875.758796     5
    143.594795     4
    198.003940     3
    5.962001       3
                  ..
    180.904395     1
    94.002828      1
    1232.877672    1
    6.408129       1
    1695.160972    1
    Name: Total_sqr_footage, Length: 1403, dtype: int64
    2.000000    232
    1.000000    224
    1.993440    208
    2.534418    200
    2.993440    163
    3.527858    126
    1.534418     99
    2.527858     68
    1.355448     28
    2.355448     21
    3.534418     11
    1.889866     11
    3.000000      8
    2.889866      7
    3.993440      6
    3.348887      5
    3.978051      5
    4.527858      4
    3.118598      3
    2.512469      2
    2.474046      2
    3.512469      2
    3.889866      2
    3.096649      2
    2.348887      2
    3.112038      1
    2.118598      1
    1.348887      1
    0.993440      1
    2.571821      1
    1.978051      1
    1.690420      1
    4.467486      1
    1.527858      1
    2.978051      1
    5.956972      1
    5.096649      1
    Name: Total_Bathrooms, dtype: int64
    0.000000     254
    31.064436     12
    9.643047      12
    9.105438      12
    38.723951      9
                ... 
    19.169684      1
    53.035518      1
    18.042164      1
    22.440714      1
    86.531959      1
    Name: Total_porch_sf, Length: 916, dtype: int64
    0    1447
    1       6
    Name: haspool, dtype: int64
    0    825
    1    628
    Name: has2ndfloor, dtype: int64
    1    1373
    0      80
    Name: hasgarage, dtype: int64
    1    1416
    0      37
    Name: hasbsmt, dtype: int64
    1    765
    0    688
    Name: hasfireplace, dtype: int64
    0    1366
    1      87
    Name: MSSubClass_120, dtype: int64
    0    1453
    Name: MSSubClass_150, dtype: int64
    0    1390
    1      63
    Name: MSSubClass_160, dtype: int64
    0    1443
    1      10
    Name: MSSubClass_180, dtype: int64
    0    1423
    1      30
    Name: MSSubClass_190, dtype: int64
    0    920
    1    533
    Name: MSSubClass_20, dtype: int64
    0    1384
    1      69
    Name: MSSubClass_30, dtype: int64
    0    1449
    1       4
    Name: MSSubClass_40, dtype: int64
    0    1441
    1      12
    Name: MSSubClass_45, dtype: int64
    0    1310
    1     143
    Name: MSSubClass_50, dtype: int64
    0    1156
    1     297
    Name: MSSubClass_60, dtype: int64
    0    1394
    1      59
    Name: MSSubClass_70, dtype: int64
    0    1437
    1      16
    Name: MSSubClass_75, dtype: int64
    0    1395
    1      58
    Name: MSSubClass_80, dtype: int64
    0    1433
    1      20
    Name: MSSubClass_85, dtype: int64
    0    1401
    1      52
    Name: MSSubClass_90, dtype: int64
    0    1445
    1       8
    Name: MSZoning_C (all), dtype: int64
    0    1388
    1      65
    Name: MSZoning_FV, dtype: int64
    0    1437
    1      16
    Name: MSZoning_RH, dtype: int64
    1    1146
    0     307
    Name: MSZoning_RL, dtype: int64
    0    1235
    1     218
    Name: MSZoning_RM, dtype: int64
    0    1403
    1      50
    Name: Alley_Grvl, dtype: int64
    1    1363
    0      90
    Name: Alley_None, dtype: int64
    0    1413
    1      40
    Name: Alley_Pave, dtype: int64
    0    972
    1    481
    Name: LotShape_IR1, dtype: int64
    0    1412
    1      41
    Name: LotShape_IR2, dtype: int64
    0    1444
    1       9
    Name: LotShape_IR3, dtype: int64
    1    922
    0    531
    Name: LotShape_Reg, dtype: int64
    0    1392
    1      61
    Name: LandContour_Bnk, dtype: int64
    0    1403
    1      50
    Name: LandContour_HLS, dtype: int64
    0    1417
    1      36
    Name: LandContour_Low, dtype: int64
    1    1306
    0     147
    Name: LandContour_Lvl, dtype: int64
    0    1192
    1     261
    Name: LotConfig_Corner, dtype: int64
    0    1359
    1      94
    Name: LotConfig_CulDSac, dtype: int64
    0    1406
    1      47
    Name: LotConfig_FR2, dtype: int64
    0    1449
    1       4
    Name: LotConfig_FR3, dtype: int64
    1    1047
    0     406
    Name: LotConfig_Inside, dtype: int64
    1    1375
    0      78
    Name: LandSlope_Gtl, dtype: int64
    0    1388
    1      65
    Name: LandSlope_Mod, dtype: int64
    0    1440
    1      13
    Name: LandSlope_Sev, dtype: int64
    0    1436
    1      17
    Name: Neighborhood_Blmngtn, dtype: int64
    0    1451
    1       2
    Name: Neighborhood_Blueste, dtype: int64
    0    1437
    1      16
    Name: Neighborhood_BrDale, dtype: int64
    0    1395
    1      58
    Name: Neighborhood_BrkSide, dtype: int64
    0    1425
    1      28
    Name: Neighborhood_ClearCr, dtype: int64
    0    1303
    1     150
    Name: Neighborhood_CollgCr, dtype: int64
    0    1402
    1      51
    Name: Neighborhood_Crawfor, dtype: int64
    0    1355
    1      98
    Name: Neighborhood_Edwards, dtype: int64
    0    1374
    1      79
    Name: Neighborhood_Gilbert, dtype: int64
    0    1418
    1      35
    Name: Neighborhood_IDOTRR, dtype: int64
    0    1436
    1      17
    Name: Neighborhood_MeadowV, dtype: int64
    0    1404
    1      49
    Name: Neighborhood_Mitchel, dtype: int64
    0    1228
    1     225
    Name: Neighborhood_NAmes, dtype: int64
    0    1444
    1       9
    Name: Neighborhood_NPkVill, dtype: int64
    0    1381
    1      72
    Name: Neighborhood_NWAmes, dtype: int64
    0    1412
    1      41
    Name: Neighborhood_NoRidge, dtype: int64
    0    1376
    1      77
    Name: Neighborhood_NridgHt, dtype: int64
    0    1340
    1     113
    Name: Neighborhood_OldTown, dtype: int64
    0    1428
    1      25
    Name: Neighborhood_SWISU, dtype: int64
    0    1380
    1      73
    Name: Neighborhood_Sawyer, dtype: int64
    0    1394
    1      59
    Name: Neighborhood_SawyerW, dtype: int64
    0    1368
    1      85
    Name: Neighborhood_Somerst, dtype: int64
    0    1428
    1      25
    Name: Neighborhood_StoneBr, dtype: int64
    0    1415
    1      38
    Name: Neighborhood_Timber, dtype: int64
    0    1442
    1      11
    Name: Neighborhood_Veenker, dtype: int64
    0    1405
    1      48
    Name: Condition1_Artery, dtype: int64
    0    1375
    1      78
    Name: Condition1_Feedr, dtype: int64
    1    1257
    0     196
    Name: Condition1_Norm, dtype: int64
    0    1445
    1       8
    Name: Condition1_PosA, dtype: int64
    0    1435
    1      18
    Name: Condition1_PosN, dtype: int64
    0    1442
    1      11
    Name: Condition1_RRAe, dtype: int64
    0    1427
    1      26
    Name: Condition1_RRAn, dtype: int64
    0    1451
    1       2
    Name: Condition1_RRNe, dtype: int64
    0    1448
    1       5
    Name: Condition1_RRNn, dtype: int64
    0    1451
    1       2
    Name: Condition2_Artery, dtype: int64
    0    1448
    1       5
    Name: Condition2_Feedr, dtype: int64
    1    1440
    0      13
    Name: Condition2_Norm, dtype: int64
    0    1452
    1       1
    Name: Condition2_PosA, dtype: int64
    0    1452
    1       1
    Name: Condition2_PosN, dtype: int64
    0    1452
    1       1
    Name: Condition2_RRAe, dtype: int64
    0    1452
    1       1
    Name: Condition2_RRAn, dtype: int64
    0    1451
    1       2
    Name: Condition2_RRNn, dtype: int64
    1    1213
    0     240
    Name: BldgType_1Fam, dtype: int64
    0    1422
    1      31
    Name: BldgType_2fmCon, dtype: int64
    0    1401
    1      52
    Name: BldgType_Duplex, dtype: int64
    0    1410
    1      43
    Name: BldgType_Twnhs, dtype: int64
    0    1339
    1     114
    Name: BldgType_TwnhsE, dtype: int64
    0    1300
    1     153
    Name: HouseStyle_1.5Fin, dtype: int64
    0    1439
    1      14
    Name: HouseStyle_1.5Unf, dtype: int64
    0    730
    1    723
    Name: HouseStyle_1Story, dtype: int64
    0    1445
    1       8
    Name: HouseStyle_2.5Fin, dtype: int64
    0    1442
    1      11
    Name: HouseStyle_2.5Unf, dtype: int64
    0    1011
    1     442
    Name: HouseStyle_2Story, dtype: int64
    0    1416
    1      37
    Name: HouseStyle_SFoyer, dtype: int64
    0    1388
    1      65
    Name: HouseStyle_SLvl, dtype: int64
    0    1440
    1      13
    Name: RoofStyle_Flat, dtype: int64
    1    1139
    0     314
    Name: RoofStyle_Gable, dtype: int64
    0    1443
    1      10
    Name: RoofStyle_Gambrel, dtype: int64
    0    1171
    1     282
    Name: RoofStyle_Hip, dtype: int64
    0    1446
    1       7
    Name: RoofStyle_Mansard, dtype: int64
    0    1451
    1       2
    Name: RoofStyle_Shed, dtype: int64
    1    1428
    0      25
    Name: RoofMatl_CompShg, dtype: int64
    0    1452
    1       1
    Name: RoofMatl_Membran, dtype: int64
    0    1452
    1       1
    Name: RoofMatl_Metal, dtype: int64
    0    1452
    1       1
    Name: RoofMatl_Roll, dtype: int64
    0    1442
    1      11
    Name: RoofMatl_Tar&Grv, dtype: int64
    0    1448
    1       5
    Name: RoofMatl_WdShake, dtype: int64
    0    1447
    1       6
    Name: RoofMatl_WdShngl, dtype: int64
    0    1433
    1      20
    Name: Exterior1st_AsbShng, dtype: int64
    0    1452
    1       1
    Name: Exterior1st_AsphShn, dtype: int64
    0    1451
    1       2
    Name: Exterior1st_BrkComm, dtype: int64
    0    1404
    1      49
    Name: Exterior1st_BrkFace, dtype: int64
    0    1452
    1       1
    Name: Exterior1st_CBlock, dtype: int64
    0    1393
    1      60
    Name: Exterior1st_CemntBd, dtype: int64
    0    1231
    1     222
    Name: Exterior1st_HdBoard, dtype: int64
    0    1452
    1       1
    Name: Exterior1st_ImStucc, dtype: int64
    0    1234
    1     219
    Name: Exterior1st_MetalSd, dtype: int64
    0    1347
    1     106
    Name: Exterior1st_Plywood, dtype: int64
    0    1451
    1       2
    Name: Exterior1st_Stone, dtype: int64
    0    1429
    1      24
    Name: Exterior1st_Stucco, dtype: int64
    0    939
    1    514
    Name: Exterior1st_VinylSd, dtype: int64
    0    1247
    1     206
    Name: Exterior1st_Wd Sdng, dtype: int64
    0    1427
    1      26
    Name: Exterior1st_WdShing, dtype: int64
    0    1433
    1      20
    Name: Exterior2nd_AsbShng, dtype: int64
    0    1450
    1       3
    Name: Exterior2nd_AsphShn, dtype: int64
    0    1446
    1       7
    Name: Exterior2nd_Brk Cmn, dtype: int64
    0    1429
    1      24
    Name: Exterior2nd_BrkFace, dtype: int64
    0    1452
    1       1
    Name: Exterior2nd_CBlock, dtype: int64
    0    1394
    1      59
    Name: Exterior2nd_CmentBd, dtype: int64
    0    1246
    1     207
    Name: Exterior2nd_HdBoard, dtype: int64
    0    1443
    1      10
    Name: Exterior2nd_ImStucc, dtype: int64
    0    1240
    1     213
    Name: Exterior2nd_MetalSd, dtype: int64
    0    1452
    1       1
    Name: Exterior2nd_Other, dtype: int64
    0    1313
    1     140
    Name: Exterior2nd_Plywood, dtype: int64
    0    1448
    1       5
    Name: Exterior2nd_Stone, dtype: int64
    0    1428
    1      25
    Name: Exterior2nd_Stucco, dtype: int64
    0    950
    1    503
    Name: Exterior2nd_VinylSd, dtype: int64
    0    1256
    1     197
    Name: Exterior2nd_Wd Sdng, dtype: int64
    0    1415
    1      38
    Name: Exterior2nd_Wd Shng, dtype: int64
    0    1438
    1      15
    Name: MasVnrType_BrkCmn, dtype: int64
    0    1010
    1     443
    Name: MasVnrType_BrkFace, dtype: int64
    1    869
    0    584
    Name: MasVnrType_None, dtype: int64
    0    1327
    1     126
    Name: MasVnrType_Stone, dtype: int64
    0    1403
    1      50
    Name: ExterQual_Ex, dtype: int64
    0    1440
    1      13
    Name: ExterQual_Fa, dtype: int64
    0    966
    1    487
    Name: ExterQual_Gd, dtype: int64
    1    903
    0    550
    Name: ExterQual_TA, dtype: int64
    0    1450
    1       3
    Name: ExterCond_Ex, dtype: int64
    0    1427
    1      26
    Name: ExterCond_Fa, dtype: int64
    0    1308
    1     145
    Name: ExterCond_Gd, dtype: int64
    0    1452
    1       1
    Name: ExterCond_Po, dtype: int64
    1    1278
    0     175
    Name: ExterCond_TA, dtype: int64
    0    1308
    1     145
    Name: Foundation_BrkTil, dtype: int64
    0    822
    1    631
    Name: Foundation_CBlock, dtype: int64
    0    809
    1    644
    Name: Foundation_PConc, dtype: int64
    0    1429
    1      24
    Name: Foundation_Slab, dtype: int64
    0    1447
    1       6
    Name: Foundation_Stone, dtype: int64
    0    1450
    1       3
    Name: Foundation_Wood, dtype: int64
    0    1335
    1     118
    Name: BsmtQual_Ex, dtype: int64
    0    1418
    1      35
    Name: BsmtQual_Fa, dtype: int64
    0    835
    1    618
    Name: BsmtQual_Gd, dtype: int64
    0    1416
    1      37
    Name: BsmtQual_None, dtype: int64
    0    808
    1    645
    Name: BsmtQual_TA, dtype: int64
    0    1409
    1      44
    Name: BsmtCond_Fa, dtype: int64
    0    1388
    1      65
    Name: BsmtCond_Gd, dtype: int64
    0    1416
    1      37
    Name: BsmtCond_None, dtype: int64
    0    1451
    1       2
    Name: BsmtCond_Po, dtype: int64
    1    1305
    0     148
    Name: BsmtCond_TA, dtype: int64
    0    1233
    1     220
    Name: BsmtExposure_Av, dtype: int64
    0    1321
    1     132
    Name: BsmtExposure_Gd, dtype: int64
    0    1339
    1     114
    Name: BsmtExposure_Mn, dtype: int64
    1    949
    0    504
    Name: BsmtExposure_No, dtype: int64
    0    1415
    1      38
    Name: BsmtExposure_None, dtype: int64
    0    1234
    1     219
    Name: BsmtFinType1_ALQ, dtype: int64
    0    1305
    1     148
    Name: BsmtFinType1_BLQ, dtype: int64
    0    1037
    1     416
    Name: BsmtFinType1_GLQ, dtype: int64
    0    1379
    1      74
    Name: BsmtFinType1_LwQ, dtype: int64
    0    1416
    1      37
    Name: BsmtFinType1_None, dtype: int64
    0    1321
    1     132
    Name: BsmtFinType1_Rec, dtype: int64
    0    1026
    1     427
    Name: BsmtFinType1_Unf, dtype: int64
    0    1434
    1      19
    Name: BsmtFinType2_ALQ, dtype: int64
    0    1421
    1      32
    Name: BsmtFinType2_BLQ, dtype: int64
    0    1439
    1      14
    Name: BsmtFinType2_GLQ, dtype: int64
    0    1407
    1      46
    Name: BsmtFinType2_LwQ, dtype: int64
    0    1415
    1      38
    Name: BsmtFinType2_None, dtype: int64
    0    1399
    1      54
    Name: BsmtFinType2_Rec, dtype: int64
    1    1250
    0     203
    Name: BsmtFinType2_Unf, dtype: int64
    0    1452
    1       1
    Name: Heating_Floor, dtype: int64
    1    1421
    0      32
    Name: Heating_GasA, dtype: int64
    0    1435
    1      18
    Name: Heating_GasW, dtype: int64
    0    1446
    1       7
    Name: Heating_Grav, dtype: int64
    0    1451
    1       2
    Name: Heating_OthW, dtype: int64
    0    1449
    1       4
    Name: Heating_Wall, dtype: int64
    1    738
    0    715
    Name: HeatingQC_Ex, dtype: int64
    0    1404
    1      49
    Name: HeatingQC_Fa, dtype: int64
    0    1213
    1     240
    Name: HeatingQC_Gd, dtype: int64
    0    1452
    1       1
    Name: HeatingQC_Po, dtype: int64
    0    1028
    1     425
    Name: HeatingQC_TA, dtype: int64
    0    1360
    1      93
    Name: CentralAir_N, dtype: int64
    1    1360
    0      93
    Name: CentralAir_Y, dtype: int64
    0    1359
    1      94
    Name: Electrical_FuseA, dtype: int64
    0    1426
    1      27
    Name: Electrical_FuseF, dtype: int64
    0    1450
    1       3
    Name: Electrical_FuseP, dtype: int64
    0    1452
    1       1
    Name: Electrical_Mix, dtype: int64
    1    1328
    0     125
    Name: Electrical_SBrkr, dtype: int64
    0    1355
    1      98
    Name: KitchenQual_Ex, dtype: int64
    0    1415
    1      38
    Name: KitchenQual_Fa, dtype: int64
    0    868
    1    585
    Name: KitchenQual_Gd, dtype: int64
    1    732
    0    721
    Name: KitchenQual_TA, dtype: int64
    0    1439
    1      14
    Name: Functional_Maj1, dtype: int64
    0    1448
    1       5
    Name: Functional_Maj2, dtype: int64
    0    1422
    1      31
    Name: Functional_Min1, dtype: int64
    0    1419
    1      34
    Name: Functional_Min2, dtype: int64
    0    1438
    1      15
    Name: Functional_Mod, dtype: int64
    0    1452
    1       1
    Name: Functional_Sev, dtype: int64
    1    1353
    0     100
    Name: Functional_Typ, dtype: int64
    0    1429
    1      24
    Name: FireplaceQu_Ex, dtype: int64
    0    1420
    1      33
    Name: FireplaceQu_Fa, dtype: int64
    0    1076
    1     377
    Name: FireplaceQu_Gd, dtype: int64
    0    765
    1    688
    Name: FireplaceQu_None, dtype: int64
    0    1434
    1      19
    Name: FireplaceQu_Po, dtype: int64
    0    1141
    1     312
    Name: FireplaceQu_TA, dtype: int64
    0    1447
    1       6
    Name: GarageType_2Types, dtype: int64
    1    867
    0    586
    Name: GarageType_Attchd, dtype: int64
    0    1434
    1      19
    Name: GarageType_Basment, dtype: int64
    0    1366
    1      87
    Name: GarageType_BuiltIn, dtype: int64
    0    1444
    1       9
    Name: GarageType_CarPort, dtype: int64
    0    1068
    1     385
    Name: GarageType_Detchd, dtype: int64
    0    1373
    1      80
    Name: GarageType_None, dtype: int64
    0    1104
    1     349
    Name: GarageFinish_Fin, dtype: int64
    0    1373
    1      80
    Name: GarageFinish_None, dtype: int64
    0    1032
    1     421
    Name: GarageFinish_RFn, dtype: int64
    0    850
    1    603
    Name: GarageFinish_Unf, dtype: int64
    0    1450
    1       3
    Name: GarageQual_Ex, dtype: int64
    0    1405
    1      48
    Name: GarageQual_Fa, dtype: int64
    0    1439
    1      14
    Name: GarageQual_Gd, dtype: int64
    0    1373
    1      80
    Name: GarageQual_None, dtype: int64
    0    1450
    1       3
    Name: GarageQual_Po, dtype: int64
    1    1305
    0     148
    Name: GarageQual_TA, dtype: int64
    0    1451
    1       2
    Name: GarageCond_Ex, dtype: int64
    0    1419
    1      34
    Name: GarageCond_Fa, dtype: int64
    0    1444
    1       9
    Name: GarageCond_Gd, dtype: int64
    0    1373
    1      80
    Name: GarageCond_None, dtype: int64
    0    1446
    1       7
    Name: GarageCond_Po, dtype: int64
    1    1321
    0     132
    Name: GarageCond_TA, dtype: int64
    0    1365
    1      88
    Name: PavedDrive_N, dtype: int64
    0    1423
    1      30
    Name: PavedDrive_P, dtype: int64
    1    1335
    0     118
    Name: PavedDrive_Y, dtype: int64
    0    1394
    1      59
    Name: Fence_GdPrv, dtype: int64
    0    1400
    1      53
    Name: Fence_GdWo, dtype: int64
    0    1298
    1     155
    Name: Fence_MnPrv, dtype: int64
    0    1442
    1      11
    Name: Fence_MnWw, dtype: int64
    1    1175
    0     278
    Name: Fence_None, dtype: int64
    0    1451
    1       2
    Name: MiscFeature_Gar2, dtype: int64
    1    1399
    0      54
    Name: MiscFeature_None, dtype: int64
    0    1451
    1       2
    Name: MiscFeature_Othr, dtype: int64
    0    1404
    1      49
    Name: MiscFeature_Shed, dtype: int64
    0    1452
    1       1
    Name: MiscFeature_TenC, dtype: int64
    0    1396
    1      57
    Name: MoSold_1, dtype: int64
    0    1366
    1      87
    Name: MoSold_10, dtype: int64
    0    1374
    1      79
    Name: MoSold_11, dtype: int64
    0    1395
    1      58
    Name: MoSold_12, dtype: int64
    0    1402
    1      51
    Name: MoSold_2, dtype: int64
    0    1347
    1     106
    Name: MoSold_3, dtype: int64
    0    1313
    1     140
    Name: MoSold_4, dtype: int64
    0    1249
    1     204
    Name: MoSold_5, dtype: int64
    0    1200
    1     253
    Name: MoSold_6, dtype: int64
    0    1220
    1     233
    Name: MoSold_7, dtype: int64
    0    1331
    1     122
    Name: MoSold_8, dtype: int64
    0    1390
    1      63
    Name: MoSold_9, dtype: int64
    0    1139
    1     314
    Name: YrSold_2006, dtype: int64
    0    1126
    1     327
    Name: YrSold_2007, dtype: int64
    0    1151
    1     302
    Name: YrSold_2008, dtype: int64
    0    1118
    1     335
    Name: YrSold_2009, dtype: int64
    0    1278
    1     175
    Name: YrSold_2010, dtype: int64
    0    1410
    1      43
    Name: SaleType_COD, dtype: int64
    0    1449
    1       4
    Name: SaleType_CWD, dtype: int64
    0    1451
    1       2
    Name: SaleType_Con, dtype: int64
    0    1445
    1       8
    Name: SaleType_ConLD, dtype: int64
    0    1448
    1       5
    Name: SaleType_ConLI, dtype: int64
    0    1448
    1       5
    Name: SaleType_ConLw, dtype: int64
    0    1334
    1     119
    Name: SaleType_New, dtype: int64
    0    1450
    1       3
    Name: SaleType_Oth, dtype: int64
    1    1264
    0     189
    Name: SaleType_WD, dtype: int64
    0    1353
    1     100
    Name: SaleCondition_Abnorml, dtype: int64
    0    1449
    1       4
    Name: SaleCondition_AdjLand, dtype: int64
    0    1441
    1      12
    Name: SaleCondition_Alloca, dtype: int64
    0    1434
    1      19
    Name: SaleCondition_Family, dtype: int64
    1    1196
    0     257
    Name: SaleCondition_Normal, dtype: int64
    0    1331
    1     122
    Name: SaleCondition_Partial, dtype: int64



```python
def overfit_reducer(df):
    """
    This function takes in a dataframe and returns a list of features that are overfitted.
    """
    overfit = []
    for i in df.columns:
        counts = df[i].value_counts()
        zeros = counts.iloc[0]
        if zeros / len(df) * 100 > 99.94:
            overfit.append(i)
    overfit = list(overfit)
    return overfit


overfitted_features = overfit_reducer(X)

X = X.drop(overfitted_features, axis=1)
X_sub = X_sub.drop(overfitted_features, axis=1)
```


```python
X.shape,y.shape, X_sub.shape
```




    ((1453, 332), (1453,), (1459, 332))



# Fitting model(simple approach)

## Train_test split
 
We have separated dependent and independent features; We have separated train and test data. So, why do we still have to split our training data? If you are curious about that, I have the answer. For this competition, when we train the machine learning algorithms, we use part of the training set, usually two-thirds of the train data. Once we train our algorithm using 2/3 of the train data, we start to test our algorithms using the remaining data. If the model performs well, we dump our test data in the algorithms to predict and submit the competition. The code below, basically splits the train data into 4 parts, <b>X_train, X_test, y_train, y_test.</b>
* <b>X_train, y_train</b> first used to train the algorithm. 
* then, **X_test** is used in that trained algorithms to predict **outcomes. **
* Once we get the **outcomes**, we compare it with **y_test**

By comparing the **outcome** of the model with **test_y**, we can determine whether our algorithms are performing well or not. Once we are confident about the result of our algorithm, we may use the model to on the original test data and submit in the challenge. I have tried to show this whole process in the visualization chart below.


```python
## Train test s
from sklearn.model_selection import train_test_split
## Train test split follows this distinguished code pattern and helps creating train and test set to build machine learning. 
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = .33, random_state = 0)
```


```python
X_train.shape, y_train.shape, X_test.shape, y_test.shape
```




    ((973, 332), (973,), (480, 332), (480,))



# Modeling the Data
 
Before modeling each algorithm, I would like to discuss them for a better understanding. This way I would review what I know and at the same time help out the community. If you already know enough about Linear Regression, you may skip this part and go straight to the part where I fit the model. However, if you take your time to read this and other model description sections and let me know how I am doing, I would genuinely appreciate it. Let's get started. 

**Linear Regression**
<div>
    We will start with one of the most basic but useful machine learning model, **Linear Regression**. However, do not let the simplicity of this model fool you, as Linear Regression is the base some of the most complex models out there. For the sake of understanding this model, we will use only two features, **SalePrice** and **GrLivArea**. Let's take a sample of the data and graph it.


```python
sample_train = previous_train.sample(300)
import seaborn as sns
plt.subplots(figsize = (15,8))
ax = plt.gca()
ax.scatter(sample_train.GrLivArea.values, sample_train.SalePrice.values, color ='b');
plt.title("Chart with Data Points");
#ax = sns.regplot(sample_train.GrLivArea.values, sample_train.SalePrice.values)
#ax.plot((sample_train.GrLivArea.values.min(),sample_train.GrLivArea.values.max()), (sample_train.SalePrice.values.mean(),sample_train.SalePrice.values.mean()), color = 'r');
```


![png](a-detailed-regression-guide-with-house-pricing_files/a-detailed-regression-guide-with-house-pricing_92_0.png)


As we discussed before, there is a linear relationship between SalePrice and GrLivArea. We want to know/estimate/predict the sale price of a house based on the given area, How do we do that? One naive way is to find the average of all the house prices. Let's find a line with the average of all houses and place it in the scatter plot. Simple enough.


```python
plt.subplots(figsize = (15,8))
ax = plt.gca()
ax.scatter(sample_train.GrLivArea.values, sample_train.SalePrice.values, color ='b');
#ax = sns.regplot(sample_train.GrLivArea.values, sample_train.SalePrice.values)
ax.plot((sample_train.GrLivArea.values.min(),sample_train.GrLivArea.values.max()), (sample_train.SalePrice.values.mean(),sample_train.SalePrice.values.mean()), color = 'r');
plt.title("Chart with Average Line");
```


![png](a-detailed-regression-guide-with-house-pricing_files/a-detailed-regression-guide-with-house-pricing_94_0.png)


You can tell this is not the most efficient way to estimate the price of houses. The average line clearly does not represent all the datapoint and fails to grasp the linear relationship between <b>GrLivArea & SalePrice. </b> Let use one of the evaluation regression metrics and find out the Mean Squared Error(more on this later) of this line.


```python
## Calculating Mean Squared Error(MSE)
sample_train['mean_sale_price'] = sample_train.SalePrice.mean()
sample_train['mse'] = np.square(sample_train.mean_sale_price - sample_train.SalePrice)
sample_train.mse.mean()
## getting mse
print("Mean Squared Error(MSE) for average line is : {}".format(sample_train.mse.mean()))
```

    Mean Squared Error(MSE) for average line is : 0.1673537968365356


> If you are reading this in my github page, you may find it difficult to follow through as the following section includes mathematical equation. Please checkout [this](https://www.kaggle.com/masumrumi/a-stats-analysis-and-ml-workflow-of-house-pricing) kernel at Kaggle. 

We will explain more on MSE later. For now, let's just say, the closer the value of MSE is to "0", the better. Of course, it makes sense since we are talking about an error(mean squared error). We want to minimize this error. How can we do that? 

Introducing **Linear Regression**, one of the most basic and straightforward models. Many of us may have learned to show the relationship between two variable using something called "y equals mX plus b." Let's refresh our memory and call upon on that equation.



# $$ {y} = mX + b $$



Here, 
* **m** = slope of the regression line. It represents the relationship between X and y. In another word, it gives weight as to for each x(horizontal space) how much y(vertical space) we have to cover. In machine learning, we call it **coefficient**. 
* **b** = y-intercept. 
* **x** and **y** are the data points located in x_axis and y_axis respectively. 


<br/>

If you would like to know more about this equation, Please check out this [video](https://www.khanacademy.org/math/algebra/two-var-linear-equations/writing-slope-intercept-equations/v/graphs-using-slope-intercept-form). 

This slope equation gives us an exact linear relationship between X and y. This relationship is "exact" because we are given X and y beforehand and based on the value of X and y, we come up with the slope and y-intercept, which in turns determine the relationship between X and y. However, in real life, data is not that simple. Often the relationship is unknown to us, and even if we know the relationship, it may not always be exact. To fit an exact slope equation in an inexact relationship of data we introduce the term error. Let's see how mathematicians express this error with the slope equation. 

## $$ y = \beta_0 + \beta_1 x + \epsilon \\ $$

And, this is the equation for a simple linear regression.
Here,
* y = Dependent variable. This is what we are trying to estimate/solve/understand. 
* $\beta_0$ = the y-intercept, it is a constant and it represents the value of y when x is 0. 
* $\beta_1$ = Slope, Weight, Coefficient of x. This metrics is the relationship between y and x. In simple terms, it shows 1 unit increase in y changes when 1 unit increases in x. 
* $x_1$ = Independent variable ( simple linear regression ) /variables.
* $ \epsilon$ = error or residual. 

### $$ \text{residual}_i = y_i - \hat{y}_i$$
This error is the only part that's different/addition from the slope equation. This error exists because in real life we will never have a dataset where the regression line crosses exactly every single data point. There will be at least a good amount of points where the regression line will not be able to go through for the sake of model specifications and ** bias-variance tradeoff **(more on this later). This error term accounts for the difference of those points. So, simply speaking, an error is the difference between an original value( $y_i$ ) and a predicted value( $\hat{y}_i$ ). 

We use this function to predict the values of one dependent(target) variable based on one independent(predictor) variable. Therefore this regression is called **Simple linear regression(SLR).** If we were to write the equation regarding the sample example above it would simply look like the following equation, 
## $$ Sale Price= \beta_0 + \beta_1 (Area) + \epsilon \\ $$

This equation gives us a line that fits the data and often performs better than the average line above. But,
* How do we know that Linear regression line is actually performing better than the average line? 
* What metrics can we use to answer that? 
* How do we know if this line is even the best line(best-fit line) for the dataset? 
* If we want to get even more clear on this we may start with answering, How do we find the $\beta_0$(intercept) and  $\beta_1$(coefficient) of the equation?

<b>Finding $\beta_0$(intercept) and  $\beta_1$(coefficient):</b>

We can use the following equation to find the $\beta_0$(intercept) and  $\beta_1$(coefficient)


### $$ \hat{\beta}_1 = r_{xy} \frac{s_y}{s_x}$$
### $$ \hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x} $$

Here...
- $\bar{y}$ : the sample mean of observed values $Y$
- $\bar{x}$ : the sample mean of observed values $X$
- $s_y$ : the sample standard deviation of observed values $Y$
- $s_x$ : the sample standard deviation of observed values $X$

    > There are two types of STD's. one is for sample population and one is for Total population.
    > Check out [this](https://statistics.laerd.com/statistical-guides/measures-of-spread-standard-deviation.php) article for more. 

- $r_{xy}$ : the sample Pearson correlation coefficient between observed $X$ and $Y$


I hope most of us know how to calculate all these components from the two equations above by hand. I am going to only mention the equation of the pearson correlation(r_xy) here as it may be unknown to some of the readers. 

### $$ r_{xy}= \frac{\sum{(x_i - \bar{x})(y_i - \bar{y})}}{\sqrt{\sum(x_i - \bar{x})^2{\sum(y_i - \bar{y})^2}}}$$

Let's get on with calculating the rest by coding.


```python
## Calculating the beta coefficients by hand. 
## mean of y. 
y_bar = sample_train.SalePrice.mean()
## mean of x. 
x_bar = sample_train.GrLivArea.mean()
## Std of y
std_y = sample_train.SalePrice.std()
## std of x
std_x = sample_train.GrLivArea.std()
## correlation of x and y
r_xy = sample_train.corr().loc['GrLivArea','SalePrice']
## finding beta_1
beta_1 = r_xy*(std_y/std_x)
## finding beta_0
beta_0 = y_bar - beta_1*x_bar
```

So, we have calculated the beta coefficients.  We can now plug them in the linear equation to get the predicted y value. Let's do that.


```python
## getting y_hat, which is the predicted y values. 
sample_train['Linear_Yhat'] = beta_0 + beta_1*sample_train['GrLivArea']
```

Now that we have our predicted y values let's see how the predicted regression line looks in the graph.


```python
# create a figure
fig = plt.figure(figsize=(15,7))
# get the axis of that figure
ax = plt.gca()

# plot a scatter plot on it with our data
ax.scatter(sample_train.GrLivArea, sample_train.SalePrice, c='b')
ax.plot(sample_train['GrLivArea'], sample_train['Linear_Yhat'], color='r');
```


![png](a-detailed-regression-guide-with-house-pricing_files/a-detailed-regression-guide-with-house-pricing_102_0.png)


Phew!! This looks like something we can work with!! Let's find out the MSE for the regression line as well.


```python
## getting mse
print("Mean Squared Error(MSE) for regression line is : {}".format(np.square(sample_train['SalePrice'] - sample_train['Linear_Yhat']).mean()))
```

    Mean Squared Error(MSE) for regression line is : 0.07957199589283931



```python
from sklearn.metrics import mean_squared_error
mean_squared_error(sample_train['SalePrice'], sample_train.Linear_Yhat)
```




    0.07957199589283928



A much-anticipated decrease in mean squared error(mse), therefore better-predicted model. The way we compare between the two predicted lines is by considering their errors. Let's put both of the model's side by side and compare the errors.


```python
## Creating a customized chart. and giving in figsize and everything. 
fig = plt.figure(constrained_layout=True, figsize=(15,5))
## creating a grid of 3 cols and 3 rows. 
grid = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
#gs = fig3.add_gridspec(3, 3)
#ax1 = fig.add_subplot(grid[row, column])
ax1 = fig.add_subplot(grid[0, :1])

# get the axis
ax1 = fig.gca()

# plot it
ax1.scatter(x=sample_train['GrLivArea'], y=sample_train['SalePrice'], c='b')
ax1.plot(sample_train['GrLivArea'], sample_train['mean_sale_price'], color='k');

# iterate over predictions
for _, row in sample_train.iterrows():
    plt.plot((row['GrLivArea'], row['GrLivArea']), (row['SalePrice'], row['mean_sale_price']), 'r-')
    
ax2 = fig.add_subplot(grid[0, 1:])

# plot it
ax2.scatter(x=sample_train['GrLivArea'], y=sample_train['SalePrice'], c='b')
ax2.plot(sample_train['GrLivArea'], sample_train['Linear_Yhat'], color='k');
# iterate over predictions
for _, row in sample_train.iterrows():
    plt.plot((row['GrLivArea'], row['GrLivArea']), (row['SalePrice'], row['Linear_Yhat']), 'r-')
```


![png](a-detailed-regression-guide-with-house-pricing_files/a-detailed-regression-guide-with-house-pricing_107_0.png)


On the two charts above, the left one is the average line, and the right one is the regression line. <font color="blue"><b>Blue</b></font> dots are observed data points and <font color="red"><b>red</b></font> lines are error distance from each observed data points to model-predicted line. As you can see, the regression line reduces much of the errors; therefore, performs much better than average line. 

Now, we need to introduce a couple of evaluation metrics that will help us compare and contrast models. One of them is mean squared error(MSE) which we used while comparing two models. Some of the other metrics are...

* RMSE (Root Mean Squared Error)
### $$ \operatorname{RMSE}= \sqrt{\frac{1}{n}\sum_{i=1}^n(\hat{y_i} - y_i)^2} $$

Here
* $y_i$ = Each observed data point. 
* $\bar{y}$ = Mean of y value.
* $\hat{y_i}$ = Predicted data point for each $x_i$ depending on i. 


* MSE(Mean Squared Error)
### $$\operatorname{MSE}= \frac{1}{n}\sum_{i=1}^n(\hat{y_i} - y_i)^2$$

* MAE (Mean Absolute Error)
### $$\operatorname{MAE} = \frac{\sum_{i=1}^n|{\bar{y} - y_i}|}{n}$$

* RSE (Relative Squared Error)
### $$\operatorname{RSE}= \frac{\sum_{i=1}^n(\hat{y_i} - y_i)^2}{\sum_{i=1}^n(\bar{y} - y_i)^2}$$

* RAE (Relative Absolute Error) 
### $$\operatorname{RAE}= \frac{\sum_{i=1}^n |\hat{y_i} - y_i|}{\sum_{i=1}^n |\bar{y} - y_i|}$$

> and 
* $R^2$ (Coefficient of the determination)



The evaluation metrics often named in such a way that I find it confusing to remember. So, this is a guide for me and everyone else who is reading it. There are many evaluation metrics. Let's name a few of them. 

It may seem confusing with multiple similar abbreviations, but once we focus on what they each do, things will become much more intuitive. For now, I am going to dive right into the $R^2$.

# $R^2$(The "Coefficient of determination"): 
> $R^2$ describes the proportion of variance of the dependent variable explained by the regression model. Let's write the equation for $R^2$. 

# $$ \operatorname{R^2} = \frac{SSR}{SST} $$

Here,

* SST(Sum of the Total Squared Error) is the total residual. It is also known as TSS(Total Sum of the Squared Error)
* SSR(Sum of the Squared Regression) is the residual explained by the regression line. SSR is also known as ESS(Explained Sum of the Squared Error)

and

* SSE(Sum of the Squared Error)/RSS(Residual Sum of the Squared Error)
Let's break these down. 

## SST/TSS:
SST is the sum of the squared distance from all points to average line ( $\bar{y}$ ). We call this the **total variation** in the Y's of the **Total Sum of the Squares(SST).** Let's see it in the function. 
### $$ \operatorname{SST} = \sum_{i=1}^n \left(y_i - \bar{y}\right)^2 $$

Here
* $y_i$ = Each observed data point. 
* $\bar{y}$ = Mean of y value.
* $\hat{y_i}$ = Predicted data point for each $x_i$ depending on i. 

A visualization would make things much more clear.
![](http://blog.hackerearth.com/wp-content/uploads/2016/12/anat.png)
 
In this visualization above, the light green line is the <font color="green"><b>average line</b></font> and the black dot is the observed value. So, SST describes the distance between the black dot and the <font color="green"><b>average line</b></font>.


## SSR/ESS:
SSR is the sum of the squared residual between each predicted value and the average line. In statistics language we say that, SSR is the squared residual explained by the regression line. In the visualization above SSR is the distance from <font color='green'><b>baseline model</b></font> to the <font color = 'blue'><b>regression line.</b></font> 
### $$ SSR = \sum_{i=1}^n \left(\hat{y_i} - \bar{y}\right)^2 $$

## SSE/RSS: 
RSS is calculated by squaring each residual of the data points and then adding them together. This residual is the difference between the predicted line and the observed value. In statistics language, we say, SSE is the squared residual that was not explained by the regression line, and this is the quantity least-square minimizes. In the chart above SSE is the distance of the actual data point from the <font color = 'blue'><b>regression line</b></font>. 

### $$ SSE = \sum_{i=1}^n \left(y_i - \hat{y}_i\right)^2 $$

And the relation between all three of these metrics is
## $$SST = SSR + SSE$$


From the equation above and the $R^2$ equation from the top we can modify the $R^2$ equation as the following
# $$ R^2 = 1 - \frac{SSE}{SST} $$

## More on $R^2$: 
* $R^2$ is matric with a value between 0 and 1. 
* If the points are perfectly linear, then error sum of squares is 0, In that case, SSR = SST. Which means the variation in the Y's is completely explained by the regression line causing the value of $R^2$ to be close to 1. 
* In other extreme cases, when there is no relation between x and y, hence SSR = 0 and therefore SSE = SST, The regression line explains none of the variances in Y causing $R^2$ to be close to 0.
* $R^2$ measures the explanatory power of the model; The more of the variance in the dependent variable(Y) the model can explain, the more powerful it is.
* $R^2$ can be infinitely negative as well. Having a negative indicates that the predictive equation has a greater error than the baseline model.
* The value of $R^2$ increases as more feature gets added despite the effectiveness of those features in the model.
* This is a problem, since we may think that having a greater $R^2$ means a better model, even though the model didnot actually improved. In order to get around this we use Adjusted R-Squared($R^2_{adj}$)

**Adjusted R-Squared($R^2_{adj}$)**: 

$R^2_{adj}$ is similar to $R^2$. However, the value of$R^2_{adj}$ decreases if we use a feature that doesn't improve the model significantly. Let's write the equation for $R^2_{adj}$. 

## $$ {R^2_{adj}} = 1 - [\frac{(1 - R^2)(n-1)}{(n-k-1)}]$$

here, 
* n = # of datapoints. 
* k = # of feature used. 

As you can see from the equation, the increase of k(feature) in the denumerator penilizes the adjusted $R^2$ value if there is not a significant improvement of $R^2$ in the numerator.  

### The following part is a work in progress!!

So, from the Evaluation section above, we know that, 
### $$ RSS = \sum_{i=1}^n \left(y_i - \hat{y}_i\right)^2 $$

And, we already know ...
## $$ \hat{y} = \beta_0 + \beta_1 x + \epsilon \\ $$

Let's plug in( $\hat{Y}$  ) equation in the RSS equation and we get...
$$RSS = \sum_{i=1}^n \left(y_i - \left(\beta_0 + \sum_{j=1}^p\beta_j x_j\right)\right)^2 $$

This equation is also known as the loss function. Here, **"loss"** is the sum of squared residuals(More on this later). 

### Mean Squared Error
Now let's get back to our naive prediction and calculate the **Mean squared error**, which is also a metrics similar to RSS, helps us determine how well our model is performing. In **Mean squared error** we subtract the mean of y from each y datapoints and square them. 


If you would like to improve this result further, you can think about the assumptions of the linear regressions and apply them as we have discussed earlier in this kernel. 


Similar to **Simple Linear Regression**, there is an equation for multiple independent variables to predict a target variable. The equation is as follows.

## $$ \hat{y} = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n $$

Here, We already know parts of the equation, and from there we can keep adding new features and their coefficients with the equations. Quite simple, isn't it. 

We can have a target variable predicted by multiple independent variables using this equation. Therefore this equation is called **Multiple Linear Regression.** Let's try this regression in the housing dataset.



```python
## importing necessary models.
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

## Call in the LinearRegression object
lin_reg = LinearRegression(normalize=True, n_jobs=-1)
## fit train and test data. 
lin_reg.fit(X_train, y_train)
## Predict test data. 
y_pred = lin_reg.predict(X_test)
```


```python
## get average squared error(MSE) by comparing predicted values with real values. 
print ('%.2f'%mean_squared_error(y_test, y_pred))
```

    3444690724184620032.00


## Using cross validation.


```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
lin_reg = LinearRegression()
cv = KFold(shuffle=True, random_state=2, n_splits=10)
scores = cross_val_score(lin_reg, X,y,cv = cv, scoring = 'neg_mean_absolute_error')
```


```python
print ('%.8f'%scores.mean())
```

    -0.23923767


 This way of model fitting above is probably the simplest way to construct a machine learning model. However, Let's dive deep into some more complex regression. 

### Regularization Models
What makes regression model more effective is its ability of *regularizing*. The term "regularizing" stands for models ability **to structurally prevent overfitting by imposing a penalty on the coefficients.** 


There are three types of regularizations. 
* **Ridge**
* **Lasso**
* **Elastic Net**

These regularization methods work by penalizing **the magnitude of the coefficients of features** and at the same time **minimizing the error between the predicted value and actual observed values**.  This minimization becomes a balance between the error (the difference between the predicted value and observed value) and the size of the coefficients. The only difference between Ridge and Lasso is **the way they penalize the coefficients.** Elastic Net is the combination of these two. **Elastic Net** adds both the sum of the squares errors and the absolute value of the squared error. To get more in-depth of it, let us review the least squared loss function. 

**Ordinary least squared** loss function minimizes the residual sum of the square(RSS) to fit the data:

### $$ \text{minimize:}\; RSS = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \sum_{i=1}^n \left(y_i - \left(\beta_0 + \sum_{j=1}^p\beta_j x_j\right)\right)^2 $$

Let's review this equation once again, Here: 
* $y_i$ is the observed value. 
* $\hat{y}_i$ is the predicted value. 
* The error = $y_i$ - $\hat{y}_i$
* The square of the error = $(y_i - \hat{y}_i)^2$
* The sum of the square of the error = $\sum_{i=1}^n (y_i - \hat{y}_i)^2$, that's the equation on the left. 
* The only difference between left sides equation vs. the right sides one above is the replacement of $\hat{y}_i$, it is replaced by $\left(\beta_0 + \sum_{j=1}^p\beta_j x_j\right)$, which simply follow's the slope equation, y = mx+b, where, 
* $\beta_0$ is the intercept. 
* **$\beta_j$ is the coefficient of the feature($x_j$).**

Let's describe the effect of regularization and then we will learn how we can use loss function in Ridge.
* One of the benefits of regularization is that it deals with **multicollinearity**(high correlation between predictor variables) well, especially Ridge method. Lasso deals with **multicollinearity** more brutally by penalizing related coefficients and force them to become zero, hence removing them. However, **Lasso** is well suited for redundant variables. 
 
***
<div>
    
 ### Ridge:
Ridge regression adds penalty equivalent to the square of the magnitude of the coefficients. This penalty is added to the least square loss function above and looks like this...

### $$ \text{minimize:}\; RSS+Ridge = \sum_{i=1}^n \left(y_i - \left(\beta_0 + \sum_{j=1}^p\beta_j x_j\right)\right)^2 + \lambda_2\sum_{j=1}^p \beta_j^2$$

Here, 
* $\lambda_2$ is constant; a regularization parameter. It is also known as $\alpha$. The higher the value of this constant the more the impact in the loss function. 
    * When $\lambda_2$ is 0, the loss funciton becomes same as simple linear regression. 
    * When $\lambda_2$ is $\infty$, the coefficients become 0
    * When $\lambda_2$ is between  0 and $\infty$(0<$\lambda_2$<$\infty$), The $\lambda_2$ parameter will decide the miagnitude given to the coefficients. The coefficients will be somewhere between 0 and ones for simple linear regression. 
* $\sum_{j=1}^p \beta_j^2$, is the squared sum of all coefficients. 

Now that we know every nitty-gritty details about this equation, let's use it for science, but before that a couple of things to remember. 
* It is essential to standardize the predictor variables before constructing the models. 
* It is important to check for multicollinearity,


```python
## Importing Ridge. 
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
## Assiging different sets of alpha values to explore which can be the best fit for the model. 
alpha_ridge = [-3,-2,-1,1e-15, 1e-10, 1e-8,1e-5,1e-4, 1e-3,1e-2,0.5,1,1.5, 2,3,4, 5, 10, 20, 30, 40]
temp_rss = {}
temp_mse = {}
for i in alpha_ridge:
    ## Assigin each model. 
    ridge = Ridge(alpha= i, normalize=True)
    ## fit the model. 
    ridge.fit(X_train, y_train)
    ## Predicting the target value based on "Test_x"
    y_pred = ridge.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rss = sum((y_pred-y_test)**2)
    temp_mse[i] = mse
    temp_rss[i] = rss
```


```python
for key, value in sorted(temp_mse.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))
```

    0.01: 0.012058094351210535
    0.001: 0.012361806588246711
    0.5: 0.012398339932154152
    0.0001: 0.01245484127382067
    1e-05: 0.012608710723829023
    1e-15: 0.012689040089985792
    1e-08: 0.012689390221530126
    1e-10: 0.012689501040831894
    1: 0.013828461512484526
    1.5: 0.015292912740672846
    2: 0.01675982655543862
    3: 0.019679216442474945
    4: 0.02256515565515432
    5: 0.025406035155256846
    10: 0.03869750083870753
    20: 0.06016951670298049
    30: 0.0759721333920899
    40: 0.08783870527373712
    -1: 22.584227365355215
    -3: 37.778420933109615
    -2: 1127.9922287334566



```python
for key, value in sorted(temp_rss.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))
```

    0.01: 5.787885288581059
    0.001: 5.93366716235842
    0.5: 5.951203167433997
    0.0001: 5.978323811433925
    1e-05: 6.0521811474379295
    1e-15: 6.09073924319318
    1e-08: 6.09090730633446
    1e-10: 6.090960499599308
    1: 6.637661525992575
    1.5: 7.340598115522962
    2: 8.04471674661054
    3: 9.446023892387972
    4: 10.831274714474079
    5: 12.194896874523279
    10: 18.574800402579594
    20: 28.881368017430614
    30: 36.466624028203164
    40: 42.16257853139385
    -1: 10840.429135370508
    -3: 18133.642047892616
    -2: 541436.2697920599


### Lasso:
Lasso adds penalty equivalent to the absolute value of the sum of coefficients. This penalty is added to the least square loss function and replaces the squared sum of coefficients from Ridge. 

## $$ \text{minimize:}\; RSS + Lasso = \sum_{i=1}^n \left(y_i - \left(\beta_0 + \sum_{j=1}^p\beta_j x_j\right)\right)^2 + \lambda_1\sum_{j=1}^p |\beta_j|$$

Here, 
* $\lambda_2$ is a constant similar to the Ridge function. 
* $\sum_{j=1}^p |\beta_j|$ is the absolute sum of the coefficients.


```python
from sklearn.linear_model import Lasso 
temp_rss = {}
temp_mse = {}
for i in alpha_ridge:
    ## Assigin each model. 
    lasso_reg = Lasso(alpha= i, normalize=True)
    ## fit the model. 
    lasso_reg.fit(X_train, y_train)
    ## Predicting the target value based on "Test_x"
    y_pred = lasso_reg.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rss = sum((y_pred-y_test)**2)
    temp_mse[i] = mse
    temp_rss[i] = rss
```


```python
for key, value in sorted(temp_mse.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))
```

    0.0001: 0.010061310720512168
    1e-05: 0.011553827897129448
    1e-08: 0.012457467856017411
    1e-10: 0.012462691942706754
    1e-15: 0.012462746144280465
    0.001: 0.01834414161633641
    0.01: 0.15998234085337285
    0.5: 0.16529633945001213
    1: 0.16529633945001213
    1.5: 0.16529633945001213
    2: 0.16529633945001213
    3: 0.16529633945001213
    4: 0.16529633945001213
    5: 0.16529633945001213
    10: 0.16529633945001213
    20: 0.16529633945001213
    30: 0.16529633945001213
    40: 0.16529633945001213
    -1: 13593671877.122139
    -2: 54374687289.91916
    -3: 122343046237.35477



```python
for key, value in sorted(temp_rss.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))
```

    0.0001: 4.829429145845844
    1e-05: 5.545837390622136
    1e-08: 5.979584570888356
    1e-10: 5.982092132499242
    1e-15: 5.982118149254621
    0.001: 8.80518797584147
    0.01: 76.79152360961896
    0.5: 79.34224293600582
    1: 79.34224293600582
    1.5: 79.34224293600582
    2: 79.34224293600582
    3: 79.34224293600582
    4: 79.34224293600582
    5: 79.34224293600582
    10: 79.34224293600582
    20: 79.34224293600582
    30: 79.34224293600582
    40: 79.34224293600582
    -1: 6524962501018.624
    -2: 26099849899161.2
    -3: 58724662193930.24


### Elastic Net: 
Elastic Net is the combination of both Ridge and Lasso. It adds both the sum of squared coefficients and the absolute sum of the coefficients with the ordinary least square function. Let's look at the function. 

### $$ \text{minimize:}\; RSS + Ridge + Lasso = \sum_{i=1}^n \left(y_i - \left(\beta_0 + \sum_{j=1}^p\beta_j x_j\right)\right)^2 + \lambda_1\sum_{j=1}^p |\beta_j| + \lambda_2\sum_{j=1}^p \beta_j^2$$

This equation is pretty self-explanatory if you have been following this kernel so far.


```python
from sklearn.linear_model import ElasticNet
temp_rss = {}
temp_mse = {}
for i in alpha_ridge:
    ## Assigin each model. 
    lasso_reg = ElasticNet(alpha= i, normalize=True)
    ## fit the model. 
    lasso_reg.fit(X_train, y_train)
    ## Predicting the target value based on "Test_x"
    y_pred = lasso_reg.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rss = sum((y_pred-y_test)**2)
    temp_mse[i] = mse
    temp_rss[i] = rss
```


```python
for key, value in sorted(temp_mse.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))
```

    0.0001: 0.010410659759083177
    1e-05: 0.011786338809652118
    1e-08: 0.012459291804442076
    1e-10: 0.012462710322589892
    1e-15: 0.012462746144464467
    0.001: 0.014970896504153937
    0.01: 0.10870286572380539
    0.5: 0.16529633945001213
    1: 0.16529633945001213
    1.5: 0.16529633945001213
    2: 0.16529633945001213
    3: 0.16529633945001213
    4: 0.16529633945001213
    5: 0.16529633945001213
    10: 0.16529633945001213
    20: 0.16529633945001213
    30: 0.16529633945001213
    40: 0.16529633945001213
    -3: 5.388825744878945
    -2: 5.4709451280041455
    -1: 5.72917572327467



```python
for key, value in sorted(temp_rss.items(), key=lambda item: item[1]):
    print("%s: %s" % (key, value))
```

    0.0001: 4.997116684359926
    1e-05: 5.657442628633018
    1e-08: 5.980460066132193
    1e-10: 5.982100954843148
    1e-15: 5.982118149342944
    0.001: 7.186030321993893
    0.01: 52.17737554742654
    0.5: 79.34224293600582
    1: 79.34224293600582
    1.5: 79.34224293600582
    2: 79.34224293600582
    3: 79.34224293600582
    4: 79.34224293600582
    5: 79.34224293600582
    10: 79.34224293600582
    20: 79.34224293600582
    30: 79.34224293600582
    40: 79.34224293600582
    -3: 2586.6363575418927
    -2: 2626.0536614419884
    -1: 2750.004347171842


# Fitting model (Advanced approach)


```python
kfolds = KFold(n_splits=10, shuffle=True, random_state=42)

def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

def cv_rmse(model, X=X):
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kfolds))
    return (rmse)
```


```python
alphas_alt = [14.5, 14.6, 14.7, 14.8, 14.9, 15, 15.1, 15.2, 15.3, 15.4, 15.5]
alphas2 = [5e-05, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008]
e_alphas = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007]
e_l1ratio = [0.8, 0.85, 0.9, 0.95, 0.99, 1]
```


```python
ridge = make_pipeline(RobustScaler(), RidgeCV(alphas=alphas_alt, cv=kfolds))
lasso = make_pipeline(RobustScaler(), LassoCV(max_iter=1e7, 
                                              alphas=alphas2, 
                                              random_state=42, 
                                              cv=kfolds))
elasticnet = make_pipeline(RobustScaler(), ElasticNetCV(max_iter=1e7, alphas=e_alphas, cv=kfolds, l1_ratio=e_l1ratio))                                
svr = make_pipeline(RobustScaler(), SVR(C= 20, epsilon= 0.008, gamma=0.0003,))
```


```python
gbr = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05, max_depth=4, max_features='sqrt', min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =42)                             
```


```python
lightgbm = LGBMRegressor(objective='regression', 
                                       num_leaves=4,
                                       learning_rate=0.01, 
                                       n_estimators=5000,
                                       max_bin=200, 
                                       bagging_fraction=0.75,
                                       bagging_freq=5, 
                                       bagging_seed=7,
                                       feature_fraction=0.2,
                                       feature_fraction_seed=7,
                                       verbose=-1,
                                       )
```


```python
xgboost = XGBRegressor(learning_rate=0.01,n_estimators=3460,
                                     max_depth=3, min_child_weight=0,
                                     gamma=0, subsample=0.7,
                                     colsample_bytree=0.7,
                                     objective='reg:linear', nthread=-1,
                                     scale_pos_weight=1, seed=27,
                                     reg_alpha=0.00006)
```


```python
stack_gen = StackingCVRegressor(regressors=(ridge, lasso, elasticnet, xgboost, lightgbm),
                                meta_regressor=xgboost,
                                use_features_in_secondary=True)
```


```python
# score = cv_rmse(stack_gen)
# print("Stack: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )
```


```python
score = cv_rmse(ridge)
print("Ridge: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(lasso)
print("LASSO: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(elasticnet)
print("elastic net: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(svr)
print("SVR: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(lightgbm)
print("lightgbm: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

# score = cv_rmse(gbr)
# print("gbr: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )

score = cv_rmse(xgboost)
print("xgboost: {:.4f} ({:.4f})\n".format(score.mean(), score.std()), datetime.now(), )
```

    Ridge: 0.1011 (0.0141)
     2020-04-17 21:57:13.160139
    LASSO: 0.0997 (0.0142)
     2020-04-17 21:57:16.920019
    elastic net: 0.0998 (0.0143)
     2020-04-17 21:57:38.961332
    SVR: 0.1020 (0.0146)
     2020-04-17 21:57:45.892114
    lightgbm: 0.1058 (0.0160)
     2020-04-17 21:57:57.648367
    [21:57:57] WARNING: /usr/local/miniconda/conda-bld/xgboost_1584539872846/work/src/objective/regression_obj.cu:167: reg:linear is now deprecated in favor of reg:squarederror.
    [21:58:20] WARNING: /usr/local/miniconda/conda-bld/xgboost_1584539872846/work/src/objective/regression_obj.cu:167: reg:linear is now deprecated in favor of reg:squarederror.
    [21:58:43] WARNING: /usr/local/miniconda/conda-bld/xgboost_1584539872846/work/src/objective/regression_obj.cu:167: reg:linear is now deprecated in favor of reg:squarederror.
    [21:59:06] WARNING: /usr/local/miniconda/conda-bld/xgboost_1584539872846/work/src/objective/regression_obj.cu:167: reg:linear is now deprecated in favor of reg:squarederror.
    [21:59:29] WARNING: /usr/local/miniconda/conda-bld/xgboost_1584539872846/work/src/objective/regression_obj.cu:167: reg:linear is now deprecated in favor of reg:squarederror.
    [21:59:52] WARNING: /usr/local/miniconda/conda-bld/xgboost_1584539872846/work/src/objective/regression_obj.cu:167: reg:linear is now deprecated in favor of reg:squarederror.
    [22:00:16] WARNING: /usr/local/miniconda/conda-bld/xgboost_1584539872846/work/src/objective/regression_obj.cu:167: reg:linear is now deprecated in favor of reg:squarederror.
    [22:00:40] WARNING: /usr/local/miniconda/conda-bld/xgboost_1584539872846/work/src/objective/regression_obj.cu:167: reg:linear is now deprecated in favor of reg:squarederror.
    [22:01:04] WARNING: /usr/local/miniconda/conda-bld/xgboost_1584539872846/work/src/objective/regression_obj.cu:167: reg:linear is now deprecated in favor of reg:squarederror.
    [22:01:28] WARNING: /usr/local/miniconda/conda-bld/xgboost_1584539872846/work/src/objective/regression_obj.cu:167: reg:linear is now deprecated in favor of reg:squarederror.
    xgboost: 0.1059 (0.0148)
     2020-04-17 22:01:52.629694



```python
print('START Fit')

print('stack_gen')
stack_gen_model = stack_gen.fit(np.array(X), np.array(y))

print('elasticnet')
elastic_model_full_data = elasticnet.fit(X, y)

print('Lasso')
lasso_model_full_data = lasso.fit(X, y)

print('Ridge') 
ridge_model_full_data = ridge.fit(X, y)

print('Svr')
svr_model_full_data = svr.fit(X, y)

# print('GradientBoosting')
# gbr_model_full_data = gbr.fit(X, y)

print('xgboost')
xgb_model_full_data = xgboost.fit(X, y)

print('lightgbm')
lgb_model_full_data = lightgbm.fit(X, y)
```

    START Fit
    stack_gen
    [22:02:52] WARNING: /usr/local/miniconda/conda-bld/xgboost_1584539872846/work/src/objective/regression_obj.cu:167: reg:linear is now deprecated in favor of reg:squarederror.
    [22:03:14] WARNING: /usr/local/miniconda/conda-bld/xgboost_1584539872846/work/src/objective/regression_obj.cu:167: reg:linear is now deprecated in favor of reg:squarederror.
    [22:03:34] WARNING: /usr/local/miniconda/conda-bld/xgboost_1584539872846/work/src/objective/regression_obj.cu:167: reg:linear is now deprecated in favor of reg:squarederror.
    [22:03:55] WARNING: /usr/local/miniconda/conda-bld/xgboost_1584539872846/work/src/objective/regression_obj.cu:167: reg:linear is now deprecated in favor of reg:squarederror.
    [22:04:15] WARNING: /usr/local/miniconda/conda-bld/xgboost_1584539872846/work/src/objective/regression_obj.cu:167: reg:linear is now deprecated in favor of reg:squarederror.
    [22:04:42] WARNING: /usr/local/miniconda/conda-bld/xgboost_1584539872846/work/src/objective/regression_obj.cu:167: reg:linear is now deprecated in favor of reg:squarederror.
    [22:05:12] WARNING: /usr/local/miniconda/conda-bld/xgboost_1584539872846/work/src/objective/regression_obj.cu:167: reg:linear is now deprecated in favor of reg:squarederror.
    elasticnet
    Lasso
    Ridge
    Svr
    xgboost
    [22:05:45] WARNING: /usr/local/miniconda/conda-bld/xgboost_1584539872846/work/src/objective/regression_obj.cu:167: reg:linear is now deprecated in favor of reg:squarederror.
    lightgbm


# Blending Models


```python
1.0 * elastic_model_full_data.predict(X)
```




    array([12.2253507 , 12.19481394, 12.28746198, ..., 12.45059163,
           11.84620006, 11.91614323])




```python
def blend_models_predict(X):
    return ((0.1 * elastic_model_full_data.predict(X)) + \
            (0.05 * lasso_model_full_data.predict(X)) + \
            (0.2 * ridge_model_full_data.predict(X)) + \
            (0.1 * svr_model_full_data.predict(X)) + \
#             (0.1 * gbr_model_full_data.predict(X)) + \
            (0.15 * xgb_model_full_data.predict(X)) + \
            (0.1 * lgb_model_full_data.predict(X)) + \
            (0.3 * stack_gen_model.predict(np.array(X))))
```


```python
print('RMSLE score on train data:')
print(rmsle(y, blend_models_predict(X)))
```

    RMSLE score on train data:
    0.06264960621850663



```python
print('Predict submission')
submission = pd.read_csv("../input/house-prices-advanced-regression-techniques/sample_submission.csv")
submission.iloc[:,1] = np.floor(np.expm1(blend_models_predict(X_sub)))
```

    Predict submission



```python
print('Blend with Top Kernels submissions\n')
sub_1 = pd.read_csv('../input/top-house-price-kernel-predictions/blending_high_scores_top_1_8th_place.csv')
sub_2 = pd.read_csv('../input/top-house-price-kernel-predictions/house_prices_ensemble_7models.csv')
sub_3 = pd.read_csv('../input/top-house-price-kernel-predictions/blend_and_stack_LR.csv')
submission.iloc[:,1] = np.floor((0.25 * np.floor(np.expm1(blend_models_predict(X_sub)))) + 
                                (0.25 * sub_1.iloc[:,1]) + 
                                (0.25 * sub_2.iloc[:,1]) + 
                                (0.25 * sub_3.iloc[:,1]))
```

    Blend with Top Kernels submissions
    


# Submission


```python
q1 = submission['SalePrice'].quantile(0.005)
q2 = submission['SalePrice'].quantile(0.995)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x > q1 else x*0.77)
submission['SalePrice'] = submission['SalePrice'].apply(lambda x: x if x < q2 else x*1.1)
submission.to_csv("submission.csv", index=False)
```

## Resources & Credits. 
* To GA where I started my data science journey.
* To Kaggle community for inspiring me over and over again with all the resources I need. 
* [Types of Standard Deviation](https://statistics.laerd.com/statistical-guides/measures-of-spread-standard-deviation.php)
* [What is Regression](https://www.youtube.com/watch?v=aq8VU5KLmkY)

![](http://)***
If you like to discuss any other projects or just have a chat about data science topics, I'll be more than happy to connect with you on:

**LinkedIn:** https://www.linkedin.com/in/masumrumi/ 

**My Website:** http://masumrumi.com/ 

*** This kernel will always be a work in progress. I will incorporate new concepts of data science as I comprehend them with each update. If you have any idea/suggestions about this notebook, please let me know. Any feedback about further improvements would be genuinely appreciated.***
***
### If you have come this far, Congratulations!!

### If this notebook helped you in any way or you liked it, please upvote and/or leave a comment!! :) 


```python

```


```python

```
