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
df=pd.DataFrame([line.split() for line in open('test/testdata/heart.dat')],columns=colnames)
df=df.astype(float)

featuremeans=df[colnames[:-1]].mean()
featurestds=df[colnames[:-1]].std()
df[colnames[:-1]]=(df[colnames[:-1]]-featuremeans)/featurestds
df.describe()

df['heartdisease']-=1
df['heartdisease'].mean()

full_model=colnames[-1]+' ~ '+' + '.join(colnames[:-1])
with pm.Model() as logistic_model:
    pm.glm.GLM.from_formula(formula=full_model,
                            data=df,
                            family=pm.glm.families.Binomial())
    
import time
start = time.time()
with logistic_model:
    trace_NUTS  = pm.sample(tune=200,
                         draws=200,
                         chains=4,
                         init = 'adapt_diag',
                         cores=3)
    
mcmcresults=pm.summary(trace_NUTS )
mcmcresults


