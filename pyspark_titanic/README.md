This kernel will give a tutorial for starting out with PySpark using Titanic dataset. Let's get started. 


### Kernel Goals
<a id="aboutthiskernel"></a>
***
There are three primary goals of this kernel.
- <b>Provide a tutorial for someone who is starting out with pyspark.
- <b>Do an exploratory data analysis(EDA)</b> of titanic with visualizations and storytelling.  
- <b>Predict</b>: Use machine learning classification models to predict the chances of passengers survival.

### What is Spark, anyway?
Spark is a platform for cluster computing. Spark lets us spread data and computations over clusters with multiple nodes (think of each node as a separate computer). Splitting up data makes it easier to work with very large datasets because each node only works with a small amount of data.
As each node works on its own subset of the total data, it also carries out a part of the total calculations required, so that both data processing and computation are performed in parallel over the nodes in the cluster. It is a fact that parallel computation can make certain types of programming tasks much faster.

Deciding whether or not Spark is the best solution for your problem takes some experience, but you can consider questions like:
* Is my data too big to work with on a single machine?
* Can my calculations be easily parallelized?




```python
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('../input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

    ../input/titanic/test.csv
    ../input/titanic/train.csv



```python
## installing pyspark
!pip install pyspark
```

    Requirement already satisfied: pyspark in /Users/masumrumi/opt/anaconda3/lib/python3.8/site-packages (3.0.0)
    Requirement already satisfied: py4j==0.10.9 in /Users/masumrumi/opt/anaconda3/lib/python3.8/site-packages (from pyspark) (0.10.9)


The first step in using Spark is connecting to a cluster. In practice, the cluster will be hosted on a remote machine that's connected to all other nodes. There will be one computer, called the master that manages splitting up the data and the computations. The master is connected to the rest of the computers in the cluster, which are called worker. The master sends the workers data and calculations to run, and they send their results back to the master.

We definitely don't need may clusters for Titanic dataset. In addition to that, the syntax for running locally or using many clusters are pretty similar. To start working with Spark DataFrames, we first have to create a SparkSession object from SparkContext. We can think of the SparkContext as the connection to the cluster and SparkSession as the interface with that connection. Let's create a SparkSession. 

# Beginner Tutorial
This part is solely for beginners. I recommend starting from to get a good understanding of the flow. 


```python
## creating a spark session
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('tutorial').getOrCreate()
```

Let's read the dataset. 


```python
df = spark.read.csv('../input/titanic/train.csv', header = True, inferSchema=True)
df_test = spark.read.csv('../input/titanic/test.csv', header = True, inferSchema=True)
```


```python
## So, what is df?
type(df)
```




    pyspark.sql.dataframe.DataFrame




```python
## As you can see it's a Spark dataframe. Let's take a look at the preview of the dataset. 
df.show()
```

    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|          Ticket|   Fare|Cabin|Embarked|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    |          1|       0|     3|Braund, Mr. Owen ...|  male|22.0|    1|    0|       A/5 21171|   7.25| null|       S|
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|        PC 17599|71.2833|  C85|       C|
    |          3|       1|     3|Heikkinen, Miss. ...|female|26.0|    0|    0|STON/O2. 3101282|  7.925| null|       S|
    |          4|       1|     1|Futrelle, Mrs. Ja...|female|35.0|    1|    0|          113803|   53.1| C123|       S|
    |          5|       0|     3|Allen, Mr. Willia...|  male|35.0|    0|    0|          373450|   8.05| null|       S|
    |          6|       0|     3|    Moran, Mr. James|  male|null|    0|    0|          330877| 8.4583| null|       Q|
    |          7|       0|     1|McCarthy, Mr. Tim...|  male|54.0|    0|    0|           17463|51.8625|  E46|       S|
    |          8|       0|     3|Palsson, Master. ...|  male| 2.0|    3|    1|          349909| 21.075| null|       S|
    |          9|       1|     3|Johnson, Mrs. Osc...|female|27.0|    0|    2|          347742|11.1333| null|       S|
    |         10|       1|     2|Nasser, Mrs. Nich...|female|14.0|    1|    0|          237736|30.0708| null|       C|
    |         11|       1|     3|Sandstrom, Miss. ...|female| 4.0|    1|    1|         PP 9549|   16.7|   G6|       S|
    |         12|       1|     1|Bonnell, Miss. El...|female|58.0|    0|    0|          113783|  26.55| C103|       S|
    |         13|       0|     3|Saundercock, Mr. ...|  male|20.0|    0|    0|       A/5. 2151|   8.05| null|       S|
    |         14|       0|     3|Andersson, Mr. An...|  male|39.0|    1|    5|          347082| 31.275| null|       S|
    |         15|       0|     3|Vestrom, Miss. Hu...|female|14.0|    0|    0|          350406| 7.8542| null|       S|
    |         16|       1|     2|Hewlett, Mrs. (Ma...|female|55.0|    0|    0|          248706|   16.0| null|       S|
    |         17|       0|     3|Rice, Master. Eugene|  male| 2.0|    4|    1|          382652| 29.125| null|       Q|
    |         18|       1|     2|Williams, Mr. Cha...|  male|null|    0|    0|          244373|   13.0| null|       S|
    |         19|       0|     3|Vander Planke, Mr...|female|31.0|    1|    0|          345763|   18.0| null|       S|
    |         20|       1|     3|Masselmani, Mrs. ...|female|null|    0|    0|            2649|  7.225| null|       C|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    only showing top 20 rows
    



```python
## It looks a bit messi. See what I did there? ;). Anyway, how about using .toPandas() for change. 
df.toPandas()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>886</th>
      <td>887</td>
      <td>0</td>
      <td>2</td>
      <td>Montvila, Rev. Juozas</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>211536</td>
      <td>13.0000</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>887</th>
      <td>888</td>
      <td>1</td>
      <td>1</td>
      <td>Graham, Miss. Margaret Edith</td>
      <td>female</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>112053</td>
      <td>30.0000</td>
      <td>B42</td>
      <td>S</td>
    </tr>
    <tr>
      <th>888</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>"Johnston, Miss. Catherine Helen ""Carrie"""</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.4500</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>889</th>
      <td>890</td>
      <td>1</td>
      <td>1</td>
      <td>Behr, Mr. Karl Howell</td>
      <td>male</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>111369</td>
      <td>30.0000</td>
      <td>C148</td>
      <td>C</td>
    </tr>
    <tr>
      <th>890</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.7500</td>
      <td>None</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>891 rows × 12 columns</p>
</div>




```python
# I use the toPandas() in a riddiculous amount as you will see in this kernel. 
# It is just convenient and doesn't put a lot of constran in my eye. 
## in addition to that if you know pandas, this can be very helpful 
## for checking your work.
## how about a summary. 
df.describe().toPandas()
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
      <th>summary</th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>count</td>
      <td>891</td>
      <td>891</td>
      <td>891</td>
      <td>891</td>
      <td>891</td>
      <td>714</td>
      <td>891</td>
      <td>891</td>
      <td>891</td>
      <td>891</td>
      <td>204</td>
      <td>889</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mean</td>
      <td>446.0</td>
      <td>0.3838383838383838</td>
      <td>2.308641975308642</td>
      <td>None</td>
      <td>None</td>
      <td>29.69911764705882</td>
      <td>0.5230078563411896</td>
      <td>0.38159371492704824</td>
      <td>260318.54916792738</td>
      <td>32.2042079685746</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>stddev</td>
      <td>257.3538420152301</td>
      <td>0.48659245426485753</td>
      <td>0.8360712409770491</td>
      <td>None</td>
      <td>None</td>
      <td>14.526497332334035</td>
      <td>1.1027434322934315</td>
      <td>0.8060572211299488</td>
      <td>471609.26868834975</td>
      <td>49.69342859718089</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>3</th>
      <td>min</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>"Andersson, Mr. August Edvard (""Wennerstrom"")"</td>
      <td>female</td>
      <td>0.42</td>
      <td>0</td>
      <td>0</td>
      <td>110152</td>
      <td>0.0</td>
      <td>A10</td>
      <td>C</td>
    </tr>
    <tr>
      <th>4</th>
      <td>max</td>
      <td>891</td>
      <td>1</td>
      <td>3</td>
      <td>van Melkebeke, Mr. Philemon</td>
      <td>male</td>
      <td>80.0</td>
      <td>8</td>
      <td>6</td>
      <td>WE/P 5735</td>
      <td>512.3292</td>
      <td>T</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
## That's better. Let's print the schema of the df using .printSchema()
df.printSchema()
```

    root
     |-- PassengerId: integer (nullable = true)
     |-- Survived: integer (nullable = true)
     |-- Pclass: integer (nullable = true)
     |-- Name: string (nullable = true)
     |-- Sex: string (nullable = true)
     |-- Age: double (nullable = true)
     |-- SibSp: integer (nullable = true)
     |-- Parch: integer (nullable = true)
     |-- Ticket: string (nullable = true)
     |-- Fare: double (nullable = true)
     |-- Cabin: string (nullable = true)
     |-- Embarked: string (nullable = true)
    



```python
# similar approach
df.dtypes
```




    [('PassengerId', 'int'),
     ('Survived', 'int'),
     ('Pclass', 'int'),
     ('Name', 'string'),
     ('Sex', 'string'),
     ('Age', 'double'),
     ('SibSp', 'int'),
     ('Parch', 'int'),
     ('Ticket', 'string'),
     ('Fare', 'double'),
     ('Cabin', 'string'),
     ('Embarked', 'string')]



The data in the real world is not this clean. We often have to create our own schema and implement it. We will describe more about it in the future. Since we are talking about schema, are you wondering if you would be able to implement sql with Spark?. Yes, you can. 

One of the best advantage of Spark is that you can run sql commands to do analysis. If you are like that nifty co-worker of mine, you would probably want to use sql with spark. Let's do an example. 


```python
## First, we need to register a sql temporary view.
df.createOrReplaceTempView("mytable");

## Then, we use spark.sql and write sql inside it. 
result = spark.sql("SELECT * FROM mytable ORDER BY Fare DESC LIMIT 10")
result.toPandas()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>738</td>
      <td>1</td>
      <td>1</td>
      <td>Lesurer, Mr. Gustave J</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>B101</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1</th>
      <td>259</td>
      <td>1</td>
      <td>1</td>
      <td>Ward, Miss. Anna</td>
      <td>female</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>None</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>680</td>
      <td>1</td>
      <td>1</td>
      <td>Cardeza, Mr. Thomas Drake Martinez</td>
      <td>male</td>
      <td>36.0</td>
      <td>0</td>
      <td>1</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>B51 B53 B55</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>342</td>
      <td>1</td>
      <td>1</td>
      <td>Fortune, Miss. Alice Elizabeth</td>
      <td>female</td>
      <td>24.0</td>
      <td>3</td>
      <td>2</td>
      <td>19950</td>
      <td>263.0000</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>89</td>
      <td>1</td>
      <td>1</td>
      <td>Fortune, Miss. Mabel Helen</td>
      <td>female</td>
      <td>23.0</td>
      <td>3</td>
      <td>2</td>
      <td>19950</td>
      <td>263.0000</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>439</td>
      <td>0</td>
      <td>1</td>
      <td>Fortune, Mr. Mark</td>
      <td>male</td>
      <td>64.0</td>
      <td>1</td>
      <td>4</td>
      <td>19950</td>
      <td>263.0000</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>6</th>
      <td>28</td>
      <td>0</td>
      <td>1</td>
      <td>Fortune, Mr. Charles Alexander</td>
      <td>male</td>
      <td>19.0</td>
      <td>3</td>
      <td>2</td>
      <td>19950</td>
      <td>263.0000</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>7</th>
      <td>312</td>
      <td>1</td>
      <td>1</td>
      <td>Ryerson, Miss. Emily Borie</td>
      <td>female</td>
      <td>18.0</td>
      <td>2</td>
      <td>2</td>
      <td>PC 17608</td>
      <td>262.3750</td>
      <td>B57 B59 B63 B66</td>
      <td>C</td>
    </tr>
    <tr>
      <th>8</th>
      <td>743</td>
      <td>1</td>
      <td>1</td>
      <td>"Ryerson, Miss. Susan Parker ""Suzette"""</td>
      <td>female</td>
      <td>21.0</td>
      <td>2</td>
      <td>2</td>
      <td>PC 17608</td>
      <td>262.3750</td>
      <td>B57 B59 B63 B66</td>
      <td>C</td>
    </tr>
    <tr>
      <th>9</th>
      <td>119</td>
      <td>0</td>
      <td>1</td>
      <td>Baxter, Mr. Quigg Edmond</td>
      <td>male</td>
      <td>24.0</td>
      <td>0</td>
      <td>1</td>
      <td>PC 17558</td>
      <td>247.5208</td>
      <td>B58 B60</td>
      <td>C</td>
    </tr>
  </tbody>
</table>
</div>




```python
# pretty cool, We will dive deep in sql later. 
# Let's go back to dataFrame and do some nitty-gritty stuff. 
# What if want the column names only. 
df.columns
```




    ['PassengerId',
     'Survived',
     'Pclass',
     'Name',
     'Sex',
     'Age',
     'SibSp',
     'Parch',
     'Ticket',
     'Fare',
     'Cabin',
     'Embarked']




```python
# What about just a column?
df['Age']
```




    Column<b'Age'>




```python
type(df['Age'])
```




    pyspark.sql.column.Column




```python
# Well, that's not what we pandas users have expected. 
# Yes, in order to get a column we need to use select().  
# df.select(df['Age']).show()
df.select('Age').show()
```

    +----+
    | Age|
    +----+
    |22.0|
    |38.0|
    |26.0|
    |35.0|
    |35.0|
    |null|
    |54.0|
    | 2.0|
    |27.0|
    |14.0|
    | 4.0|
    |58.0|
    |20.0|
    |39.0|
    |14.0|
    |55.0|
    | 2.0|
    |null|
    |31.0|
    |null|
    +----+
    only showing top 20 rows
    



```python
## What if we want multiple columns?
df.select(['Age', 'Fare']).show()
```

    +----+-------+
    | Age|   Fare|
    +----+-------+
    |22.0|   7.25|
    |38.0|71.2833|
    |26.0|  7.925|
    |35.0|   53.1|
    |35.0|   8.05|
    |null| 8.4583|
    |54.0|51.8625|
    | 2.0| 21.075|
    |27.0|11.1333|
    |14.0|30.0708|
    | 4.0|   16.7|
    |58.0|  26.55|
    |20.0|   8.05|
    |39.0| 31.275|
    |14.0| 7.8542|
    |55.0|   16.0|
    | 2.0| 29.125|
    |null|   13.0|
    |31.0|   18.0|
    |null|  7.225|
    +----+-------+
    only showing top 20 rows
    



```python
# that's more like it. 
# What about accessing a row
df.head(1)
```




    [Row(PassengerId=1, Survived=0, Pclass=3, Name='Braund, Mr. Owen Harris', Sex='male', Age=22.0, SibSp=1, Parch=0, Ticket='A/5 21171', Fare=7.25, Cabin=None, Embarked='S')]




```python
type(df.head(1))
```




    list




```python
## returns a list. let's get the item in the list
row = df.head(1)[0]
row
```




    Row(PassengerId=1, Survived=0, Pclass=3, Name='Braund, Mr. Owen Harris', Sex='male', Age=22.0, SibSp=1, Parch=0, Ticket='A/5 21171', Fare=7.25, Cabin=None, Embarked='S')




```python
type(row)
```




    pyspark.sql.types.Row




```python
## row can be converted into dict using .asDict()
row.asDict()
```




    {'PassengerId': 1,
     'Survived': 0,
     'Pclass': 3,
     'Name': 'Braund, Mr. Owen Harris',
     'Sex': 'male',
     'Age': 22.0,
     'SibSp': 1,
     'Parch': 0,
     'Ticket': 'A/5 21171',
     'Fare': 7.25,
     'Cabin': None,
     'Embarked': 'S'}




```python
## Then the value can be accessed from the row dictionaly. 
row.asDict()['PassengerId']
```




    1




```python
## similarly
row.asDict()['Name']
```




    'Braund, Mr. Owen Harris'




```python
## let's say we want to change the name of a column. we can use withColumnRenamed
# df.withColumnRenamed('exsisting name', 'anticipated name');
df.withColumnRenamed("Age", "newA").limit(5).toPandas()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>newA</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>None</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Let's say we want to modify a column add in this case, adding $20 with every fare. 
## df.withColumn('existing column', 'calculation with the column(we have to put df not just column)')
## so not df.withColumn('Fare', 'Fare' +20).show()
df.withColumn('Fare', df['Fare']+20).limit(5).show()
```

    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|          Ticket|   Fare|Cabin|Embarked|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    |          1|       0|     3|Braund, Mr. Owen ...|  male|22.0|    1|    0|       A/5 21171|  27.25| null|       S|
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|        PC 17599|91.2833|  C85|       C|
    |          3|       1|     3|Heikkinen, Miss. ...|female|26.0|    0|    0|STON/O2. 3101282| 27.925| null|       S|
    |          4|       1|     1|Futrelle, Mrs. Ja...|female|35.0|    1|    0|          113803|   73.1| C123|       S|
    |          5|       0|     3|Allen, Mr. Willia...|  male|35.0|    0|    0|          373450|  28.05| null|       S|
    +-----------+--------+------+--------------------+------+----+-----+-----+----------------+-------+-----+--------+
    



```python
## let's say we want get the average fare.
# we will use the "mean" function from pyspark.sql.functions(this is where all the functions are stored) and
# collect the data using ".collect()" instead of using .show()
# collect returns a list so we need to get the value from the list using index
```


```python
from pyspark.sql.functions import mean
fare_mean = df.select(mean("Fare")).collect()
fare_mean[0][0]
```




    32.2042079685746




```python
fare_mean = fare_mean[0][0]
```


```python
# What if we want to filter data and see all datapoints above average. 
# there are two approaches of this, we can use sql syntex
# or just dataframe approach. 
df.filter("Fare > 32.20" ).limit(3).show()
```

    +-----------+--------+------+--------------------+------+----+-----+-----+--------+-------+-----+--------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|  Ticket|   Fare|Cabin|Embarked|
    +-----------+--------+------+--------------------+------+----+-----+-----+--------+-------+-----+--------+
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|PC 17599|71.2833|  C85|       C|
    |          4|       1|     1|Futrelle, Mrs. Ja...|female|35.0|    1|    0|  113803|   53.1| C123|       S|
    |          7|       0|     1|McCarthy, Mr. Tim...|  male|54.0|    0|    0|   17463|51.8625|  E46|       S|
    +-----------+--------+------+--------------------+------+----+-----+-----+--------+-------+-----+--------+
    



```python
## using the dataframe approach
df.filter(df['Fare']> fare_mean).limit(3).show()
```

    +-----------+--------+------+--------------------+------+----+-----+-----+--------+-------+-----+--------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|  Ticket|   Fare|Cabin|Embarked|
    +-----------+--------+------+--------------------+------+----+-----+-----+--------+-------+-----+--------+
    |          2|       1|     1|Cumings, Mrs. Joh...|female|38.0|    1|    0|PC 17599|71.2833|  C85|       C|
    |          4|       1|     1|Futrelle, Mrs. Ja...|female|35.0|    1|    0|  113803|   53.1| C123|       S|
    |          7|       0|     1|McCarthy, Mr. Tim...|  male|54.0|    0|    0|   17463|51.8625|  E46|       S|
    +-----------+--------+------+--------------------+------+----+-----+-----+--------+-------+-----+--------+
    



```python
## What if we want to filter by multiple columns.
# passenger with below average fare with a Pclass equals 3
df.filter((df['Fare'] < fare_mean) &
          (df['Pclass'] ==  3)
         ).toPandas()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6</td>
      <td>0</td>
      <td>3</td>
      <td>Moran, Mr. James</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>330877</td>
      <td>8.4583</td>
      <td>None</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>4</th>
      <td>8</td>
      <td>0</td>
      <td>3</td>
      <td>Palsson, Master. Gosta Leonard</td>
      <td>male</td>
      <td>2.0</td>
      <td>3</td>
      <td>1</td>
      <td>349909</td>
      <td>21.0750</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>456</th>
      <td>883</td>
      <td>0</td>
      <td>3</td>
      <td>Dahlberg, Miss. Gerda Ulrika</td>
      <td>female</td>
      <td>22.0</td>
      <td>0</td>
      <td>0</td>
      <td>7552</td>
      <td>10.5167</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>457</th>
      <td>885</td>
      <td>0</td>
      <td>3</td>
      <td>Sutehall, Mr. Henry Jr</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>SOTON/OQ 392076</td>
      <td>7.0500</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>458</th>
      <td>886</td>
      <td>0</td>
      <td>3</td>
      <td>Rice, Mrs. William (Margaret Norton)</td>
      <td>female</td>
      <td>39.0</td>
      <td>0</td>
      <td>5</td>
      <td>382652</td>
      <td>29.1250</td>
      <td>None</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>459</th>
      <td>889</td>
      <td>0</td>
      <td>3</td>
      <td>"Johnston, Miss. Catherine Helen ""Carrie"""</td>
      <td>female</td>
      <td>NaN</td>
      <td>1</td>
      <td>2</td>
      <td>W./C. 6607</td>
      <td>23.4500</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>460</th>
      <td>891</td>
      <td>0</td>
      <td>3</td>
      <td>Dooley, Mr. Patrick</td>
      <td>male</td>
      <td>32.0</td>
      <td>0</td>
      <td>0</td>
      <td>370376</td>
      <td>7.7500</td>
      <td>None</td>
      <td>Q</td>
    </tr>
  </tbody>
</table>
<p>461 rows × 12 columns</p>
</div>




```python
# passenger with below fare and are not male
df.filter((df['Fare'] < fare_mean) &
          ~(df['Sex'] ==  "male")
         ).limit(5).toPandas()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>1</td>
      <td>3</td>
      <td>Johnson, Mrs. Oscar W (Elisabeth Vilhelmina Berg)</td>
      <td>female</td>
      <td>27.0</td>
      <td>0</td>
      <td>2</td>
      <td>347742</td>
      <td>11.1333</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>1</td>
      <td>2</td>
      <td>Nasser, Mrs. Nicholas (Adele Achem)</td>
      <td>female</td>
      <td>14.0</td>
      <td>1</td>
      <td>0</td>
      <td>237736</td>
      <td>30.0708</td>
      <td>None</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11</td>
      <td>1</td>
      <td>3</td>
      <td>Sandstrom, Miss. Marguerite Rut</td>
      <td>female</td>
      <td>4.0</td>
      <td>1</td>
      <td>1</td>
      <td>PP 9549</td>
      <td>16.7000</td>
      <td>G6</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12</td>
      <td>1</td>
      <td>1</td>
      <td>Bonnell, Miss. Elizabeth</td>
      <td>female</td>
      <td>58.0</td>
      <td>0</td>
      <td>0</td>
      <td>113783</td>
      <td>26.5500</td>
      <td>C103</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Let's group by Pclass and see how the average fare price. 
df.groupBy("Pclass").mean().toPandas()
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
      <th>Pclass</th>
      <th>avg(PassengerId)</th>
      <th>avg(Survived)</th>
      <th>avg(Pclass)</th>
      <th>avg(Age)</th>
      <th>avg(SibSp)</th>
      <th>avg(Parch)</th>
      <th>avg(Fare)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>461.597222</td>
      <td>0.629630</td>
      <td>1.0</td>
      <td>38.233441</td>
      <td>0.416667</td>
      <td>0.356481</td>
      <td>84.154687</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>439.154786</td>
      <td>0.242363</td>
      <td>3.0</td>
      <td>25.140620</td>
      <td>0.615071</td>
      <td>0.393075</td>
      <td>13.675550</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>445.956522</td>
      <td>0.472826</td>
      <td>2.0</td>
      <td>29.877630</td>
      <td>0.402174</td>
      <td>0.380435</td>
      <td>20.662183</td>
    </tr>
  </tbody>
</table>
</div>




```python
## let's just look at the Pclass and avg(Fare)
df.groupBy("Pclass").mean().select(['Pclass', 'avg(Fare)']).show()
```

    +------+------------------+
    |Pclass|         avg(Fare)|
    +------+------------------+
    |     1| 84.15468749999992|
    |     3|13.675550101832997|
    |     2| 20.66218315217391|
    +------+------------------+
    



```python
## What if we want just the average of all fare, we can use .agg with the dataframe. 
df.agg({'Fare':'mean'}).show()
```

    +----------------+
    |       avg(Fare)|
    +----------------+
    |32.2042079685746|
    +----------------+
    



```python
## another way this can be done is by importing "mean" funciton from pyspark.sql.functions
from pyspark.sql.functions import mean
df.select(mean("Fare")).show()
```

    +----------------+
    |       avg(Fare)|
    +----------------+
    |32.2042079685746|
    +----------------+
    



```python
## we can also combine the few previous approaches to get similar results. 
temp = df.groupBy("Pclass")
temp.agg({"Fare": 'mean'}).show()
```

    +------+------------------+
    |Pclass|         avg(Fare)|
    +------+------------------+
    |     1| 84.15468749999992|
    |     3|13.675550101832997|
    |     2| 20.66218315217391|
    +------+------------------+
    



```python
# What if we want to format the results. 
# for example,
# I want to rename the column. this will be accomplished using .alias() method.  
# I want to format the number with only two decimals. this can be done using "format_number"
from pyspark.sql.functions import format_number
temp = df.groupBy("Pclass")
temp = temp.agg({"Fare": 'mean'})
temp.select('Pclass', format_number("avg(Fare)", 2).alias("average fare")).show()
```

    +------+------------+
    |Pclass|average fare|
    +------+------------+
    |     1|       84.15|
    |     3|       13.68|
    |     2|       20.66|
    +------+------------+
    



```python
## What if I want to order by Fare in ascending order. 
df.orderBy("Fare").limit(20).toPandas()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>180</td>
      <td>0</td>
      <td>3</td>
      <td>Leonard, Mr. Lionel</td>
      <td>male</td>
      <td>36.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0000</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>482</td>
      <td>0</td>
      <td>2</td>
      <td>"Frost, Mr. Anthony Wood ""Archie"""</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239854</td>
      <td>0.0000</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>414</td>
      <td>0</td>
      <td>2</td>
      <td>Cunningham, Mr. Alfred Fleming</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239853</td>
      <td>0.0000</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>303</td>
      <td>0</td>
      <td>3</td>
      <td>Johnson, Mr. William Cahoone Jr</td>
      <td>male</td>
      <td>19.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0000</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>264</td>
      <td>0</td>
      <td>1</td>
      <td>Harrison, Mr. William</td>
      <td>male</td>
      <td>40.0</td>
      <td>0</td>
      <td>0</td>
      <td>112059</td>
      <td>0.0000</td>
      <td>B94</td>
      <td>S</td>
    </tr>
    <tr>
      <th>5</th>
      <td>675</td>
      <td>0</td>
      <td>2</td>
      <td>Watson, Mr. Ennis Hastings</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239856</td>
      <td>0.0000</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>6</th>
      <td>733</td>
      <td>0</td>
      <td>2</td>
      <td>Knight, Mr. Robert J</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239855</td>
      <td>0.0000</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>7</th>
      <td>807</td>
      <td>0</td>
      <td>1</td>
      <td>Andrews, Mr. Thomas Jr</td>
      <td>male</td>
      <td>39.0</td>
      <td>0</td>
      <td>0</td>
      <td>112050</td>
      <td>0.0000</td>
      <td>A36</td>
      <td>S</td>
    </tr>
    <tr>
      <th>8</th>
      <td>816</td>
      <td>0</td>
      <td>1</td>
      <td>Fry, Mr. Richard</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>112058</td>
      <td>0.0000</td>
      <td>B102</td>
      <td>S</td>
    </tr>
    <tr>
      <th>9</th>
      <td>823</td>
      <td>0</td>
      <td>1</td>
      <td>Reuchlin, Jonkheer. John George</td>
      <td>male</td>
      <td>38.0</td>
      <td>0</td>
      <td>0</td>
      <td>19972</td>
      <td>0.0000</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>10</th>
      <td>467</td>
      <td>0</td>
      <td>2</td>
      <td>Campbell, Mr. William</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239853</td>
      <td>0.0000</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>11</th>
      <td>272</td>
      <td>1</td>
      <td>3</td>
      <td>Tornquist, Mr. William Henry</td>
      <td>male</td>
      <td>25.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0000</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>12</th>
      <td>598</td>
      <td>0</td>
      <td>3</td>
      <td>Johnson, Mr. Alfred</td>
      <td>male</td>
      <td>49.0</td>
      <td>0</td>
      <td>0</td>
      <td>LINE</td>
      <td>0.0000</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>13</th>
      <td>278</td>
      <td>0</td>
      <td>2</td>
      <td>"Parkes, Mr. Francis ""Frank"""</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>239853</td>
      <td>0.0000</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>14</th>
      <td>634</td>
      <td>0</td>
      <td>1</td>
      <td>Parr, Mr. William Henry Marsh</td>
      <td>male</td>
      <td>NaN</td>
      <td>0</td>
      <td>0</td>
      <td>112052</td>
      <td>0.0000</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>15</th>
      <td>379</td>
      <td>0</td>
      <td>3</td>
      <td>Betros, Mr. Tannous</td>
      <td>male</td>
      <td>20.0</td>
      <td>0</td>
      <td>0</td>
      <td>2648</td>
      <td>4.0125</td>
      <td>None</td>
      <td>C</td>
    </tr>
    <tr>
      <th>16</th>
      <td>873</td>
      <td>0</td>
      <td>1</td>
      <td>Carlsson, Mr. Frans Olof</td>
      <td>male</td>
      <td>33.0</td>
      <td>0</td>
      <td>0</td>
      <td>695</td>
      <td>5.0000</td>
      <td>B51 B53 B55</td>
      <td>S</td>
    </tr>
    <tr>
      <th>17</th>
      <td>327</td>
      <td>0</td>
      <td>3</td>
      <td>Nysveen, Mr. Johan Hansen</td>
      <td>male</td>
      <td>61.0</td>
      <td>0</td>
      <td>0</td>
      <td>345364</td>
      <td>6.2375</td>
      <td>None</td>
      <td>S</td>
    </tr>
    <tr>
      <th>18</th>
      <td>844</td>
      <td>0</td>
      <td>3</td>
      <td>Lemberopolous, Mr. Peter L</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>2683</td>
      <td>6.4375</td>
      <td>None</td>
      <td>C</td>
    </tr>
    <tr>
      <th>19</th>
      <td>819</td>
      <td>0</td>
      <td>3</td>
      <td>Holm, Mr. John Fredrik Alexander</td>
      <td>male</td>
      <td>43.0</td>
      <td>0</td>
      <td>0</td>
      <td>C 7075</td>
      <td>6.4500</td>
      <td>None</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
## What about descending order
df.orderBy(df['Fare'].desc()).limit(5).toPandas()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>259</td>
      <td>1</td>
      <td>1</td>
      <td>Ward, Miss. Anna</td>
      <td>female</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>None</td>
      <td>C</td>
    </tr>
    <tr>
      <th>1</th>
      <td>680</td>
      <td>1</td>
      <td>1</td>
      <td>Cardeza, Mr. Thomas Drake Martinez</td>
      <td>male</td>
      <td>36.0</td>
      <td>0</td>
      <td>1</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>B51 B53 B55</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>738</td>
      <td>1</td>
      <td>1</td>
      <td>Lesurer, Mr. Gustave J</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>PC 17755</td>
      <td>512.3292</td>
      <td>B101</td>
      <td>C</td>
    </tr>
    <tr>
      <th>3</th>
      <td>439</td>
      <td>0</td>
      <td>1</td>
      <td>Fortune, Mr. Mark</td>
      <td>male</td>
      <td>64.0</td>
      <td>1</td>
      <td>4</td>
      <td>19950</td>
      <td>263.0000</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>89</td>
      <td>1</td>
      <td>1</td>
      <td>Fortune, Miss. Mabel Helen</td>
      <td>female</td>
      <td>23.0</td>
      <td>3</td>
      <td>2</td>
      <td>19950</td>
      <td>263.0000</td>
      <td>C23 C25 C27</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>




```python
## How do we deal with missing values. 
# df.na.drop(how=("any"/"all"), thresh=(1,2,3,4,5...))
df.na.drop(how="any").limit(5).toPandas()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
      <th>name_length</th>
      <th>nLength_group</th>
      <th>title</th>
      <th>family_size</th>
      <th>family_group</th>
      <th>is_alone</th>
      <th>calculated_fare</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>G</td>
      <td>S</td>
      <td>23</td>
      <td>medium</td>
      <td>Mr</td>
      <td>1</td>
      <td>loner</td>
      <td>1</td>
      <td>7.2500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>C</td>
      <td>51</td>
      <td>long</td>
      <td>Mrs</td>
      <td>1</td>
      <td>loner</td>
      <td>1</td>
      <td>71.2833</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>G</td>
      <td>S</td>
      <td>22</td>
      <td>medium</td>
      <td>Miss</td>
      <td>0</td>
      <td>loner</td>
      <td>1</td>
      <td>7.9250</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>C</td>
      <td>S</td>
      <td>44</td>
      <td>good</td>
      <td>Mrs</td>
      <td>1</td>
      <td>loner</td>
      <td>1</td>
      <td>53.1000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>G</td>
      <td>S</td>
      <td>24</td>
      <td>medium</td>
      <td>Mr</td>
      <td>0</td>
      <td>loner</td>
      <td>1</td>
      <td>8.0500</td>
    </tr>
  </tbody>
</table>
</div>



# Advanced Tutorial
## Dealing with Missing Values
### Cabin


```python
# filling the null values in cabin with "N".
# df.fillna(value, subset=[]);
df = df.na.fill('N', subset=['Cabin'])
df_test = df_test.na.fill('N', subset=['Cabin'])
```

### Fare


```python
## how do we find out the rows with missing values?
# we can use .where(condition) with .isNull()
df_test.where(df_test['Fare'].isNull()).show()
```

    +-----------+------+------------------+----+----+-----+-----+------+----+-----+--------+
    |PassengerId|Pclass|              Name| Sex| Age|SibSp|Parch|Ticket|Fare|Cabin|Embarked|
    +-----------+------+------------------+----+----+-----+-----+------+----+-----+--------+
    |       1044|     3|Storey, Mr. Thomas|male|60.5|    0|    0|  3701|null|    N|       S|
    +-----------+------+------------------+----+----+-----+-----+------+----+-----+--------+
    


Here, We can take the average of the **Fare** column to fill in the NaN value. However, for the sake of learning and practicing, we will try something else. We can take the average of the values where **Pclass** is ***3***, **Sex** is ***male*** and **Embarked** is ***S***


```python
missing_value = df_test.filter(
    (df_test['Pclass'] == 3) &
    (df_test.Embarked == 'S') &
    (df_test.Sex == "male")
)
## filling in the null value in the fare column using Fare mean. 
df_test = df_test.na.fill(
    missing_value.select(mean('Fare')).collect()[0][0],
    subset=['Fare']
)
```


```python
# Checking
df_test.where(df_test['Fare'].isNull()).show()
```

    +-----------+------+----+---+---+-----+-----+------+----+-----+--------+
    |PassengerId|Pclass|Name|Sex|Age|SibSp|Parch|Ticket|Fare|Cabin|Embarked|
    +-----------+------+----+---+---+-----+-----+------+----+-----+--------+
    +-----------+------+----+---+---+-----+-----+------+----+-----+--------+
    


### Embarked


```python
df.where(df['Embarked'].isNull()).show()
```

    +-----------+--------+------+--------------------+------+----+-----+-----+------+----+-----+--------+
    |PassengerId|Survived|Pclass|                Name|   Sex| Age|SibSp|Parch|Ticket|Fare|Cabin|Embarked|
    +-----------+--------+------+--------------------+------+----+-----+-----+------+----+-----+--------+
    |         62|       1|     1| Icard, Miss. Amelie|female|38.0|    0|    0|113572|80.0|  B28|    null|
    |        830|       1|     1|Stone, Mrs. Georg...|female|62.0|    0|    0|113572|80.0|  B28|    null|
    +-----------+--------+------+--------------------+------+----+-----+-----+------+----+-----+--------+
    



```python
## Replacing the null values in the Embarked column with the mode. 
df = df.na.fill('C', subset=['Embarked'])
```


```python
## checking
df.where(df['Embarked'].isNull()).show()
```

    +-----------+--------+------+----+---+---+-----+-----+------+----+-----+--------+
    |PassengerId|Survived|Pclass|Name|Sex|Age|SibSp|Parch|Ticket|Fare|Cabin|Embarked|
    +-----------+--------+------+----+---+---+-----+-----+------+----+-----+--------+
    +-----------+--------+------+----+---+---+-----+-----+------+----+-----+--------+
    



```python
df_test.where(df_test.Embarked.isNull()).show()
```

    +-----------+------+----+---+---+-----+-----+------+----+-----+--------+
    |PassengerId|Pclass|Name|Sex|Age|SibSp|Parch|Ticket|Fare|Cabin|Embarked|
    +-----------+------+----+---+---+-----+-----+------+----+-----+--------+
    +-----------+------+----+---+---+-----+-----+------+----+-----+--------+
    


## Feature Engineering
### Cabin


```python
## this is a code to create a wrapper for function, that works for both python and Pyspark.
from typing import Callable
from pyspark.sql import Column
from pyspark.sql.functions import udf, col
from pyspark.sql.types import StringType, IntegerType, ArrayType, DataType
class py_or_udf:
    def __init__(self, returnType : DataType=StringType()):
        self.spark_udf_type = returnType
        
    def __call__(self, func : Callable):
        def wrapped_func(*args, **kwargs):
            if any([isinstance(arg, Column) for arg in args]) or \
                any([isinstance(vv, Column) for vv in kwargs.values()]):
                return udf(func, self.spark_udf_type)(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapped_func

    
@py_or_udf(returnType=StringType())
def first_char(col):
    return col[0]
    
```


```python
df = df.withColumn('Cabin', first_char(df['Cabin']))
```


```python
df_test = df_test.withColumn('Cabin', first_char(df_test['Cabin']))
```


```python
df.limit(5).toPandas()
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
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>N</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>N</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>N</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



We can use the average of the fare column We can use pyspark's ***groupby*** function to get the mean fare of each cabin letter.


```python
df.groupBy('Cabin').mean("Fare").show()
```

    +-----+------------------+
    |Cabin|         avg(Fare)|
    +-----+------------------+
    |    F| 18.69679230769231|
    |    E|46.026693749999986|
    |    T|              35.5|
    |    B|113.50576382978724|
    |    D| 57.24457575757576|
    |    C|100.15134067796612|
    |    A|39.623886666666664|
    |    N|  19.1573253275109|
    |    G|          13.58125|
    +-----+------------------+
    


Now, these mean can help us determine the unknown cabins, if we compare each unknown cabin rows with the given mean's above. Let's write a simple function so that we can give cabin names based on the means. 


```python
@py_or_udf(returnType=StringType())
def cabin_estimator(i):
    """Grouping cabin feature by the first letter"""
    a = 0
    if i<16:
        a = "G"
    elif i>=16 and i<27:
        a = "F"
    elif i>=27 and i<38:
        a = "T"
    elif i>=38 and i<47:
        a = "A"
    elif i>= 47 and i<53:
        a = "E"
    elif i>= 53 and i<54:
        a = "D"
    elif i>=54 and i<116:
        a = 'C'
    else:
        a = "B"
    return a
```


```python
## separating data where Cabin == 'N', remeber we used 'N' for Null. 
df_withN = df.filter(df['Cabin'] == 'N')
df2 = df.filter(df['Cabin'] != 'N')

## replacing 'N' using cabin estimated function. 
df_withN = df_withN.withColumn('Cabin', cabin_estimator(df_withN['Fare']))

# putting the dataframe back together. 
df = df_withN.union(df2).orderBy('PassengerId') 
```


```python
#let's do the same for test set
df_testN = df_test.filter(df_test['Cabin'] == 'N')
df_testNoN = df_test.filter(df_test['Cabin'] != 'N')
df_testN = df_testN.withColumn('Cabin', cabin_estimator(df_testN['Fare']))
df_test = df_testN.union(df_testNoN).orderBy('PassengerId')
```

### Name


```python
## creating UDF functions
@py_or_udf(returnType=IntegerType())
def name_length(name):
    return len(name)


@py_or_udf(returnType=StringType())
def name_length_group(size):
    a = ''
    if (size <=20):
        a = 'short'
    elif (size <=35):
        a = 'medium'
    elif (size <=45):
        a = 'good'
    else:
        a = 'long'
    return a
```


```python
## getting the name length from name. 
df = df.withColumn("name_length", name_length(df['Name']))

## grouping based on name length. 
df = df.withColumn("nLength_group", name_length_group(df['name_length']))
```


```python
## Let's do the same for test set. 
df_test = df_test.withColumn("name_length", name_length(df_test['Name']))

df_test = df_test.withColumn("nLength_group", name_length_group(df_test['name_length']))
```

### Title


```python
## this function helps getting the title from the name. 
@py_or_udf(returnType=StringType())
def get_title(name):
    return name.split('.')[0].split(',')[1].strip()

df = df.withColumn("title", get_title(df['Name']))
df_test = df_test.withColumn('title', get_title(df_test['Name']))
```


```python
## we are writing a function that can help us modify title column
@py_or_udf(returnType=StringType())
def fuse_title1(feature):
    """
    This function helps modifying the title column
    """
    if feature in ['the Countess','Capt','Lady','Sir','Jonkheer','Don','Major','Col', 'Rev', 'Dona', 'Dr']:
        return 'rare'
    elif feature in ['Ms', 'Mlle']:
        return 'Miss'
    elif feature == 'Mme':
        return 'Mrs'
    else:
        return feature
```


```python
df = df.withColumn("title", fuse_title1(df["title"]))
```


```python
df_test = df_test.withColumn("title", fuse_title1(df_test['title']))
```


```python
print(df.toPandas()['title'].unique())
print(df_test.toPandas()['title'].unique())
```

    ['Mr' 'Mrs' 'Miss' 'Master' 'rare']
    ['Mr' 'Mrs' 'Miss' 'Master' 'rare']


### family_size


```python
df = df.withColumn("family_size", df['SibSp']+df['Parch'])
df_test = df_test.withColumn("family_size", df_test['SibSp']+df_test['Parch'])
```


```python
## bin the family size. 
@py_or_udf(returnType=StringType())
def family_group(size):
    """
    This funciton groups(loner, small, large) family based on family size
    """
    
    a = ''
    if (size <= 1):
        a = 'loner'
    elif (size <= 4):
        a = 'small'
    else:
        a = 'large'
    return a
```


```python
df = df.withColumn("family_group", family_group(df['family_size']))
df_test = df_test.withColumn("family_group", family_group(df_test['family_size']))

```

### is_alone


```python
@py_or_udf(returnType=IntegerType())
def is_alone(num):
    if num<2:
        return 1
    else:
        return 0
```


```python
df = df.withColumn("is_alone", is_alone(df['family_size']))
df_test = df_test.withColumn("is_alone", is_alone(df_test["family_size"]))
```

### ticket


```python
## dropping ticket column
df = df.drop('ticket')
df_test = df_test.drop("ticket")
```

### calculated_fare


```python
from pyspark.sql.functions import expr, col, when, coalesce, lit
```


```python
## here I am using a something similar to if and else statement, 
#when(condition, value_when_condition_met).otherwise(alt_condition)
df = df.withColumn(
    "calculated_fare", 
    when((col("Fare")/col("family_size")).isNull(), col('Fare')).otherwise((col("Fare")/col("family_size"))))
```


```python
df_test = df_test.withColumn(
    "calculated_fare", 
    when((col("Fare")/col("family_size")).isNull(), col('Fare')).otherwise((col("Fare")/col("family_size"))))

```

### fare_group


```python
@py_or_udf(returnType=StringType())
def fare_group(fare):
    """
    This function creates a fare group based on the fare provided
    """
    
    a= ''
    if fare <= 4:
        a = 'Very_low'
    elif fare <= 10:
        a = 'low'
    elif fare <= 20:
        a = 'mid'
    elif fare <= 45:
        a = 'high'
    else:
        a = "very_high"
    return a
```


```python
df = df.withColumn("fare_group", fare_group(col("Fare")))
df_test = df_test.withColumn("fare_group", fare_group(col("Fare")))
```

# That's all for today. Let's come back tomorrow when we will learn how to apply machine learning with Pyspark

<div class="alert alert-info">
    <h1>Resources</h1>
    <ul>
        <li><a href="https://docs.databricks.com/spark/latest/spark-sql/udf-python.html">User-defined functions - Python</a></li>
        <li><a href="https://medium.com/@ayplam/developing-pyspark-udfs-d179db0ccc87">Developing PySpark UDFs</a></li>
    </ul>
        <h1>Credits</h1>
    <ul>
        <li>To DataCamp, I have learned so much from DataCamp.</li>
        <li>To Jose Portilla, Such an amazing teacher with all of his resources</li>
    </ul>
    
</div>

<div class="alert alert-info">
<h4>If you like to discuss any other projects or just have a chat about data science topics, I'll be more than happy to connect with you on:</h4>
    <ul>
        <li><a href="https://www.linkedin.com/in/masumrumi/"><b>LinkedIn</b></a></li>
        <li><a href="https://github.com/masumrumi"><b>Github</b></a></li>
        <li><a href="https://masumrumi.com/"><b>masumrumi.com</b></a></li>
    </ul>

<p>This kernel will always be a work in progress. I will incorporate new concepts of data science as I comprehend them with each update. If you have any idea/suggestions about this notebook, please let me know. Any feedback about further improvements would be genuinely appreciated.</p>

<h1>If you have come this far, Congratulations!!</h1>

<h1>If this notebook helped you in any way or you liked it, please upvote and/or leave a comment!! :)</h1></div>


```python

```
