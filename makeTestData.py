import warnings
import pymc3 as pm
import pandas as pd
import arviz as az
import theano

warnings.filterwarnings('ignore')
theano.config.compute_test_value = "warn"


colnames=['age','sex','chestpain','restbp','cholestoral',
          'bsugar','electrocardiographic','maxhr','angina',
          'oldpeak','slopepexercise','numvessels','thal',
          'heartdisease'
         ]

#https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/
#df=pd.DataFrame([line.split() for line in open('test/testdata/heart.dat')],columns=colnames)
df=pd.DataFrame([line.split() for line in open('data/raw/heart.dat')],columns=colnames)
df=df.astype(float)
#Generate test data
for n in df[colnames[:-1]]:
    if not n in ['age','maxhr']:
        df[n]=0.0
df.to_csv('test/testdata/heart.csv', index = False)
