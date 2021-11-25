import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pickle
from collections import OrderedDict
import pandas as pd
import numpy as np
import scipy
from scipy import stats

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import (roc_curve, roc_auc_score, confusion_matrix, accuracy_score, f1_score, 
                             precision_recall_curve) 

import theano
import pymc3 as pm
from pymc3.variational.callbacks import CheckParametersConvergence
import statsmodels.formula.api as smf

import arviz as az
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import seaborn as sns
from IPython.display import HTML


warnings.filterwarnings('ignore')
theano.config.compute_test_value = "warn"

import logging

logging.basicConfig(filename='app.log', 
                    filemode='w',
                    format='%(asctime)s - %(name)s %(levelname)s %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.DEBUG)

import sys
print('Number of arguments:', len(sys.argv), 'arguments.')
print( 'Argument List:', str(sys.argv))
testmode=False
if sys.argv[-1].lower().strip()=='test':
    logging.info('App started in test mode. Using test data (generated with just 2 features. See makeTestData.py)')
    testmode=True
else:
    logging.info('App started in full mode.')
    


colnames=['age','sex','chestpain','restbp','cholestoral',
          'bsugar','electrocardiographic','maxhr','angina',
          'oldpeak','slopepexercise','numvessels','thal',
          'heartdisease'
         ]

#https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/
if testmode:
    df=pd.read_csv('test/testdata/heart.csv')
else:
    df=pd.DataFrame([line.split() for line in open('data/raw/heart.dat')],columns=colnames)

df=df.astype(float)

logging.info('Reading data\n'+(df.describe().__str__()))

#Removing zero sd columns (eg for test data)
df=df.loc[:,df.std()!=0]
colnames=df.columns

featuremeans=df[colnames[:-1]].mean()
featurestds=df[colnames[:-1]].std()
df[colnames[:-1]]=(df[colnames[:-1]]-featuremeans)/featurestds
df['heartdisease']-=1
df['heartdisease'].mean()

logging.info('Normalized data\n'+(df.describe().__str__()))

plt.figure(figsize=(13, 13))
corr = df.corr() 
mask = np.tri(*corr.shape).T 
sns.heatmap(corr, mask=mask, annot=True)
b, t = plt.ylim() 
b += 0.5 
t -= 0.5 
plt.ylim(b, t) 
plt.savefig('corr_heat.png')
logging.info('Created corelation heatmap as corr_heat.png')

full_model=colnames[-1]+' ~ '+' + '.join(colnames[:-1])
with pm.Model() as logistic_model:
    pm.glm.GLM.from_formula(formula=full_model,
                            data=df,
                            family=pm.glm.families.Binomial())
logging.info('Created logistic regression model.')  
    
import time
start = time.time()

draws=10000
if testmode:
    draws=100
    
with logistic_model:
    trace_NUTS  = pm.sample(tune=200,
                         draws=draws,
                         chains=4,
                         init = 'adapt_diag',
                         cores=3)
    
mcmcresults=pm.summary(trace_NUTS )
logging.info('NUTS done in {}s'.format(time.time()-start)+'\n'+(mcmcresults.__str__()))

az.plot_pair(trace_NUTS, figsize=(10, 10));
plt.savefig('nuts_plot_pair.png')

#ADVI
start = time.time()
with logistic_model:
    callback = CheckParametersConvergence(diff='absolute')
    approx = pm.fit(n=100000, callbacks=[callback])
trace_advi = approx.sample(2000)
adviruntime = time.time()-start
adviresults=pm.summary(trace_advi)
logging.info('ADVI done in {}s'.format(adviruntime)+'\n'+(adviresults.__str__()))
az.plot_pair(trace_advi, figsize=(10, 10));
plt.savefig('advi_plot_pair.png')

mu_star=mcmcresults['mean'].values
tracedf=pm.trace_to_dataframe(trace_NUTS )
covactual=tracedf.cov().values
oactual=np.linalg.inv(covactual)
d=len(mu_star)



def omega(u0,U,Λ):
    return np.identity(d)*u0**2 + np.matmul(np.matmul(U,Λ),U.T)

def omega2(oactual,U,Λ):
    return np.diag(oactual)*np.identity(d) + np.matmul(np.matmul(U,Λ),U.T)

def phi(theta):
    return 8.5-scipy.stats.multivariate_normal.logpdf(theta, mean=mu_star, cov=covactual, allow_singular=False)

import scipy
import scipy.stats as st
from collections import defaultdict


FNByMethod=defaultdict(list)
iter=2000
if testmode:
    iter=100
for demeanphi in [True, False]:
    for demeantheta in [True, False]:
        p=d
        Λ = np.identity(p)
        U=np.eye(d,p)
        u0=1
        u0t,Ut,Λt=u0,U,Λ

        for t in range(iter):

            u0t=1.0
            om=omega(u0t,Ut,Λt) #as per pdf 165
            thiscov=np.linalg.inv(om)

            #Sample as per equation 6
            num_samples=10+t
            #num_samples=1000
            samples=(np.random.multivariate_normal(mu_star,thiscov,num_samples))
            phis=phi(samples)
            if demeanphi:
                phis-=phis.mean()
            expphi=np.abs(phis).mean()        

            ht=0.01/(1+expphi)
            ht*=500/(t+500)  

            t2sum=0
            denom=0
            for thisphi,thetajt in zip(phis,samples):
                if demeantheta:
                    thetajt=(thetajt-mu_star).reshape((d,1))
                else:
                    thetajt=(thetajt).reshape((d,1))
                #original formula
                t2sum+=np.matmul((np.matmul(np.matmul(-om,thetajt),thetajt.T)+np.identity(d)),np.matmul(Ut,Λt)) *thisphi
                #simplified formula
                #t2sum+=np.matmul((np.matmul(np.matmul(-om,thetajt),thetajt.T)),np.matmul(Ut,Λt)) *thisphi 
                denom+=1
            if denom:
                t2sum/=denom
                #original formula
                Ut=(Ut-ht*np.matmul(Ut,Λt)-ht*t2sum)
                #simplified formula
                #Ut=Ut-ht*t2sum 

            #update as per equation 8
            q, r = scipy.linalg.qr(Ut,mode='economic')
            Ut=q

            if t%10==0: 
                #updating eigenvalues as per pdf 191
                om=omega(u0t,Ut,np.identity(p)) #as per pdf 165
                for i in range(p):
                    ui=Ut[:,i]
                    Λt[i][i]=np.matmul(np.matmul(ui.T,oactual),ui)-1
                om=omega(u0t,Ut,Λt) #as per pdf 165

                #suggested by mentor: Frobenius Norm
                fn=np.linalg.norm(oactual-om)
                FNByMethod[(demeanphi,demeantheta)].append(fn)
                print('demeanphi={} demeantheta={} epoach {} avg phi {:.4f} Frobenius Norm {:.4f}'.format(demeanphi,demeantheta,t,expphi,fn))
                
                
fig = plt.figure()
ax = plt.subplot(111)
for demeanphi in [True, False]:
    for demeantheta in [True, False]:
        ax.plot(range(len(FNByMethod[(demeanphi,demeantheta)])),np.log(FNByMethod[(demeanphi,demeantheta)]),label='demeanphi={} demeantheta={}'.format(demeanphi,demeantheta))
    
plt.ylabel('log(FN)')
plt.xlabel('epoch')
plt.legend(bbox_to_anchor = (1.05, 0.6))
plt.title('log(FN)')

plt.savefig('capstone_vi_methods.png')
logging.info('capstone_vi_methods done')

