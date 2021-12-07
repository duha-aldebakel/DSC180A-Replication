import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pickle
from collections import OrderedDict
import pandas as pd
import numpy as np
import scipy
from scipy import stats
from collections import defaultdict

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

import plotly.graph_objects as go

import scipy.stats as st
from collections import defaultdict

import plotly.io as pio
png_renderer = pio.renderers["png"]
png_renderer.width = 1000
png_renderer.height = 500

pio.renderers.default = "png"


warnings.filterwarnings('ignore')
theano.config.compute_test_value = "warn"

import logging

logging.basicConfig(filename='app.log', 
                    filemode='w',
                    format='%(asctime)s - %(name)s %(levelname)s %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S',
                    level=logging.DEBUG)

import sys
logging.info('Number of arguments: {} arguments.'.format(len(sys.argv)))
logging.info( 'Argument List: {}'.format(str(sys.argv)))
testmode=False
if sys.argv[-1].lower().strip()=='test':
    logging.info('App started in test mode. Using test data (generated with just 2 features. See makeTestData.py)')
    testmode=True
else:
    logging.info('App started in full mode.')
    
'''
Statlog (Heart) Data Set
https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/

Attribute Information:
  -- 1. age       
  -- 2. sex       
  -- 3. chest pain type  (4 values)       
  -- 4. resting blood pressure  
  -- 5. serum cholestoral in mg/dl      
  -- 6. fasting blood sugar > 120 mg/dl       
  -- 7. resting electrocardiographic results  (values 0,1,2) 
  -- 8. maximum heart rate achieved  
  -- 9. exercise induced angina    
  -- 10. oldpeak = ST depression induced by exercise relative to rest   
  -- 11. the slope of the peak exercise ST segment     
  -- 12. number of major vessels (0-3) colored by flourosopy        
  -- 13.  thal: 3 = normal; 6 = fixed defect; 7 = reversable defect     
Final column: Absence (1) or presence (2) of heart disease

(I will change it to 1 for HD and 0 for none later on)
'''

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

'''
EXPLORATORY ANALYSIS
'''


plt.figure(figsize=(13, 13))
corr = df.corr() 
mask = np.tri(*corr.shape).T 
sns.heatmap(corr, mask=mask, annot=True)
b, t = plt.ylim() 
b += 0.5 
t -= 0.5 
plt.ylim(b, t) 
plt.savefig('images/corr_heat.png')
plt.close()

logging.info('Created corelation heatmap as corr_heat.png')

n_fts = len(df.columns)
colors = cm.rainbow(np.linspace(0, 1, n_fts))

df.drop('heartdisease',axis=1).corrwith(df.heartdisease).sort_values(ascending=True).plot(kind='barh', 
                                                                                     color=colors, figsize=(12, 6))
plt.title('Correlation to Target (heartdisease)')
plt.savefig('images/CorrelationToTarget.png')
plt.close()

logging.info('\n{}'.format(df.drop('heartdisease',axis=1).corrwith(df.heartdisease).sort_values(ascending=False)))


'''
MAPS
'''


full_model=colnames[-1]+' ~ '+' + '.join(colnames[:-1])
with pm.Model() as logistic_model:
    pm.glm.GLM.from_formula(formula=full_model,
                            data=df,
                            family=pm.glm.families.Binomial())
logging.info('Created logistic regression model.')  
    
    

'''
NUTS
'''

import time
start = time.time()

draws=2000
if testmode:
    draws=100
    
with logistic_model:
    trace_NUTS  = pm.sample(tune=draws,
                         draws=draws,
                         chains=3,
                         init = 'adapt_diag',
                         cores=3)
mcmcruntime = time.time()-start
mcmcresults=pm.summary(trace_NUTS )
logging.info('NUTS done in {}s'.format(time.time()-start)+'\n'+(mcmcresults.__str__()))

az.plot_pair(trace_NUTS, figsize=(10, 10));
plt.savefig('images/nuts_plot_pair.png')
plt.close()

tracedf=pm.trace_to_dataframe(trace_NUTS )
covactual=tracedf.cov().values
oactual=np.linalg.inv(covactual)
mu_star=mcmcresults['mean'].values
d=len(mu_star)


## Varying NUTS samples to get statistical performance vs time graph
### Performance is measured on precision matrix Omega against long run NUTS.
NUTStimeToFN={}
for numsamples in [20*i+10 for i in range(5)]+[20*i+10 for i in range(10)]+[500*i+500 for i in range(5)]+[5000*i+5000 for i in range(1)]:
    start=time.time()
    with logistic_model:
        trace_NUTS2 = pm.sample(tune=numsamples,
                             draws=numsamples,
                             chains=2,
                             init = 'adapt_diag',
                             cores=1)
        #Note: for apples-to-apples comparision, we use only one core since the same 
        #will be done for the other methods
    rt=time.time()-start
    tracedf2=pm.trace_to_dataframe(trace_NUTS2)
    cov2=tracedf2.cov().values
    o2=np.linalg.inv(cov2)
    fn=np.linalg.norm(oactual-o2)
    NUTStimeToFN[rt]=fn
    logging.info('NUTS Runtime={:.2f} FN={:.2f}'.format(rt,fn))

XY=sorted(zip(NUTStimeToFN.keys(),NUTStimeToFN.values()))
X=np.array([x[0] for x in XY])
y=np.array([np.log(x[1]) for x in XY])
plt.xlabel('Runtime(s)')
plt.ylabel('log(Frobenius Norm) vs LT NUTS')
plt.scatter(X, y,c="red",label='NUTS')

params,_=scipy.optimize.curve_fit(lambda t,a,b,c: a+b*(1/np.clip(t+c,0.0001,None)),  X,  y)
a=params[0]
b=params[1]
c=params[2]
def fit(t):
    return a+b*(1/np.clip(t+c,0.0001,None))
plt.plot(np.arange(min(X),max(X)), fit(np.arange(min(X),max(X))), 'r',c="blue",label='NUTS')
plt.legend()
plt.savefig('images/nuts_tradeoff.png')
plt.close()

'''
Frequentist
'''
start = time.time()
from sklearn.linear_model import LogisticRegression
XData=df[colnames[:-1]].values
yData=df[colnames[-1]].values
clf = LogisticRegression().fit(XData, yData)
#From https://stats.stackexchange.com/questions/89484/how-to-compute-the-standard-errors-of-a-logistic-regressions-coefficients

# Calculate matrix of predicted class probabilities.
# Check resLogit.classes_ to make sure that sklearn ordered your classes as expected
predProbs = clf.predict_proba(XData)


# Design matrix -- add column of 1's at the beginning of your X_train matrix
X_design = np.hstack([np.ones((XData.shape[0], 1)), XData])

# While V should be a diagonal matrix of size (n,n), we keep it as (n,1) here to save space.
V = np.product(predProbs, axis=1)

# Covariance matrix
om = np.dot(X_design.T* V, X_design)
covLogit = np.linalg.pinv(om)

# Standard errors
logging.info("Coefs: {}".format(clf.coef_))
logging.info("Intercept: {}".format(clf.intercept_))
logging.info("Standard errors: {}".format(np.sqrt(np.diag(covLogit))))
logging.info("Average Standard errors: {}".format(np.mean(np.sqrt(np.diag(covLogit)))))
u0est=np.sqrt(np.diag(om)).min()
logging.info("Average u0 = {}".format(u0est))

sklearnruntime = time.time()-start
accuracy=clf.score(XData, yData)
logging.info('SKlearn Logistic Regression Accuracy ={:.3f} Runtime(s) = {:.2f}'.format(accuracy,sklearnruntime))

sklearnom=np.linalg.inv(covLogit)
sklearnfn=np.linalg.norm(oactual-sklearnom)
logging.info('For SKLearn, FN ={} log(FN)={}'.format(sklearnfn,np.log(sklearnfn)))




'''
ADVI
'''
ADVItimeToFN={}
for nums in [5000*(x+1) for x in range(10)]+[1000*(x+1) for x in range(10)]:
    start = time.time()
    with logistic_model:
        callback = CheckParametersConvergence(diff='absolute')
        approx = pm.fit(n=nums, callbacks=[callback])
    trace_advi = approx.sample(2000)
    samples=[]
    for v in trace_advi._straces[0].samples.values():
        samples.append(v)
    samples=np.array(samples)
    covadvi=np.cov(samples)
    oadvi=np.linalg.inv(covadvi)
    fn=np.linalg.norm(oactual-oadvi)
    adviruntime=time.time()-start
    ADVItimeToFN[adviruntime]=fn
    
trace_advi = approx.sample(2000)
adviresults=pm.summary(trace_advi)
logging.info('ADVI done \n'+(adviresults.__str__()))
az.plot_pair(trace_advi, figsize=(10, 10));
plt.savefig('images/advi_plot_pair.png')
plt.close()
   
    
print(ADVItimeToFN)
plt.xlabel('Runtime(s)')
plt.ylabel('log(Frobenius Norm) vs LT NUTS')
maxX=40

for res,l,col in [(ADVItimeToFN,'ADVI','blue'),(NUTStimeToFN,'NUTS','red')]:
    XY=sorted(zip(res.keys(),res.values()))
    X=np.array([x[0] for x in XY])
    y=np.array([np.log(x[1]) for x in XY])
    plt.scatter(X[X<maxX], y[X<maxX],c=col,label=l)
    #fit,param=np.polynomial.polynomial.Polynomial.fit(X,y, 6, full=True)
    #params,_=scipy.optimize.curve_fit(lambda t,a,b,c: a+b*np.log(t-c**2),  X,  y)
    try:
        params,_=scipy.optimize.curve_fit(lambda t,a,b,c: a+b*(1/np.clip(t+c,0.0001,None)),  X,  y)
        a=params[0]
        b=params[1]
        c=params[2]
        def fit(t):
            return a+b*(1/np.clip(t+c,0.0001,None))
        plotrange=np.arange(min(X),min(max(X)*1.1,maxX))
        plt.plot(plotrange, fit(plotrange), 'r',c=col,label=l)
    except:
        continue
plt.legend()
plt.savefig('images/advi_nuts_tradeoff.png')
plt.close()



fig = go.Figure()

maxX=40
for res,l,col in [(ADVItimeToFN,'ADVI','blue'),(NUTStimeToFN,'NUTS','red')]:
    XY=sorted(zip(res.keys(),res.values()))
    X=np.array([x[0] for x in XY])
    y=np.array([np.log(x[1]) for x in XY])
    trace = go.Scatter(x=X[X<maxX], y=y[X<maxX],mode='markers',name=l)
    fig.add_trace(trace)
    #fit,param=np.polynomial.polynomial.Polynomial.fit(X,y, 6, full=True)
    #params,_=scipy.optimize.curve_fit(lambda t,a,b,c: a+b*np.log(t-c**2),  X,  y)
    
    params,_=scipy.optimize.curve_fit(lambda t,a,b,c: a+b*(1/np.clip(t+c,0.0001,None)),  X,  y)
    a=params[0]
    b=params[1]
    c=params[2]
    def fit(t):
        return a+b*(1/np.clip(t+c,0.0001,None))
    plotrange=np.arange(min(X),min(max(X)*1.1,maxX))
    trace = go.Scatter(x=plotrange, y=fit(plotrange),name=l)
    
    fig.add_trace(trace)

fig.update_layout(title='Computational and Statistical Tradeoffs')
fig.update_xaxes(title="Runtime(s)")
fig.update_yaxes(title="log(Frobenius Norm) vs LT NUTS")
fig.write_image('images/advi_nuts_tradeoff2.png')



def omega(u0,U,Λ):
    return np.identity(d)*u0**2 + np.matmul(np.matmul(U,Λ),U.T)

def omega2(oactual,U,Λ):
    return np.diag(oactual)*np.identity(d) + np.matmul(np.matmul(U,Λ),U.T)

def phi(theta):
    if len(theta.shape)>1:
        ps=[]
        for t in theta:
            ps.append(phi(t))
        return np.array(ps)
    clf.intercept_[0]=theta[0]
    clf.coef_[0]=theta[1:]
    return ((clf.predict_proba(XData)[:,1]-yData)**2).sum()

def getSamples(mu_star,om,num_samples): #Sample as per equation 6
    thiscov=np.linalg.inv(om)
    samples=(np.random.multivariate_normal(mu_star,thiscov,num_samples))
    phis=phi(samples)
    return samples,phis

## Fixing mathematical/performance defects in capstone pdf
### Question about demeaning variables (phi,theta)



FNByMethod=defaultdict(list)

for demeanphi in [True, False]:
    for demeantheta in [True, False]:
        p=14
        Λ = np.identity(p)
        U=np.eye(d,p)
        u0=1
        u0t,Ut,Λt=u0,U,Λ
        start=time.time()

        for t in range(2000):

            om=omega(u0t,Ut,Λt) #as per pdf 165

            num_samples=1000
            samples,phis=getSamples(mu_star,om,num_samples) #Sample as per equation 6
                        
            if demeanphi:
                phis-=phis.mean()
            expphi=np.abs(phis).mean()        

            ht=0.01/(1+expphi)

            t2sum=0
            for thisphi,thetajt in zip(phis,samples):
                if demeantheta:
                    thetajt=(thetajt-mu_star).reshape((d,1))
                else:
                    thetajt=(thetajt).reshape((d,1))
                t2sum+=np.matmul((np.matmul(np.matmul(-om,thetajt),thetajt.T)+np.identity(d)),np.matmul(Ut,Λt)) *thisphi
            t2sum/=num_samples
            Ut=(Ut-ht*np.matmul(Ut,Λt)-ht*t2sum)

            #update as per equation 8
            Ut, r = scipy.linalg.qr(Ut,mode='economic')

            if t%10==0: 
                #updating eigenvalues as per pdf 191
                om=omega(u0t,Ut,np.identity(p)) #as per pdf 165
                for i in range(p):
                    ui=Ut[:,i]
                    Λt[i][i]=np.matmul(np.matmul(ui.T,oactual),ui)-1
                    
                om=omega(u0t,Ut,Λt) #as per pdf 165
                fn=np.linalg.norm(oactual-om) #suggested by mentor: Frobenius Norm
                estart=time.time()-start
                FNByMethod[(demeanphi,demeantheta)].append((estart,fn))
                if estart>60:
                    break
                logging.info('demeanphi={} demeantheta={} epoach {} avg phi {:.4f} Frobenius Norm {:.4f}'.format(demeanphi,demeantheta,t,expphi,fn))
                
                
        
fig = go.Figure()

for demeanphi in [True, False]:
    for demeantheta in [True, False]:
        XY=FNByMethod[(demeanphi,demeantheta)]
        y=np.log(np.array([y for x,y in XY]))
        x=np.array([x for x,y in XY])
        trace = go.Scatter(x=x, y=y, 
                           name='demeanphi={} demeantheta={}'.format(demeanphi,demeantheta))
        fig.add_trace(trace)

fig.update_layout(title='Logistic Regression (Heart Dataset) Methodologies')
fig.update_xaxes(title="Runtime(s)")
fig.update_yaxes(title="log(Frobenius Norm) vs LT NUTS")
fig.update_layout(
    legend=dict(
        x=0,
        y=-0.5,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
        ),
    )
)
fig.write_image('images/demeanpsitheta.png')

## Question about sample size
###(Answer is smaller first and bigger later.)

FNByMethod3=defaultdict(list)

for num_samples2 in [0,200,400,800]:
    p=14
    Λ = np.identity(p)
    U=np.eye(d,p)
    u0=1
    u0t,Ut,Λt=u0,U,Λ
    start=time.time()

    for t in range(10000):
        om=omega(u0t,Ut,Λt) #as per pdf 165
        
        num_samples=num_samples2
        if not num_samples:
            num_samples=200+t*3

        samples,phis=getSamples(mu_star,om,num_samples) #Sample as per equation 6
        phis-=phis.mean()
        expphi=np.abs(phis).mean()        

        ht=0.01/(1+expphi)

        t2sum=0
        for thisphi,thetajt in zip(phis,samples):
            thetajt=(thetajt-mu_star).reshape((d,1))
            t2sum+=np.matmul((np.matmul(np.matmul(-om,thetajt),thetajt.T)+np.identity(d)),np.matmul(Ut,Λt)) *thisphi
        t2sum/=num_samples
        Ut=(Ut-ht*np.matmul(Ut,Λt)-ht*t2sum)

        #update as per equation 8
        Ut, r = scipy.linalg.qr(Ut,mode='economic')

        if t%10==0: 
            #updating eigenvalues as per pdf 191
            om=omega(u0t,Ut,np.identity(p)) #as per pdf 165
            for i in range(p):
                ui=Ut[:,i]
                Λt[i][i]=np.matmul(np.matmul(ui.T,oactual),ui)-1

            om=omega(u0t,Ut,Λt) #as per pdf 165
            #suggested by mentor: Frobenius Norm
            fn=np.linalg.norm(oactual-om)
            estart=time.time()-start
            FNByMethod3[(num_samples2)].append((estart,fn))
            if estart>30:
                break
            logging.info('num_samples={} epoach {} avg phi {:.4f} Frobenius Norm {:.4f}'.format(num_samples,t,expphi,fn))

fig = go.Figure()

for num_samples in [0,200,400,800]:
    XY=FNByMethod3[(num_samples)]
    y=np.log(np.array([y for x,y in XY]))
    x=np.array([x for x,y in XY])
    name='num_samples={}'.format(num_samples)
    if not num_samples:
        name='num_samples=range(200,1000)'
    trace = go.Scatter(x=x, y=y, 
                       name=name)
    fig.add_trace(trace)

fig.update_layout(title='Question about sample sizes')
fig.update_xaxes(title="Runtime(s)")
fig.update_yaxes(title="log(Frobenius Norm) vs LT NUTS")
fig.update_layout(
    legend=dict(
        x=0,
        y=-0.5,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
        ),
    )
)
fig.write_image('images/samplesize.png')

## Question about value of u0
### We can bootstrap of u0est from the SKlearn point estimate of standard error.
logging.info('We can bootstrap of u0est from the SKlearn point estimate of standard error. u0est = {}'.format(u0est))

bestu0={}
for p in [2,14]:
    FNRT_By_u0=defaultdict(list)
    bestfb=np.inf
    for u0 in [1,2,3,4,5]:
        Λ = np.identity(p)
        U=np.eye(d,p)
        u0t,Ut,Λt=u0,U,Λ
        start=time.time()

        for t in range(10000):
            om=omega(u0t,Ut,Λt) #as per pdf 165

            num_samples=200+t*3
            samples,phis=getSamples(mu_star,om,num_samples) #Sample as per equation 6
            phis-=phis.mean()
            expphi=np.abs(phis).mean()        

            ht=0.01/(1+expphi)

            t2sum=0
            for thisphi,thetajt in zip(phis,samples):
                thetajt=(thetajt-mu_star).reshape((d,1))
                t2sum+=np.matmul((np.matmul(np.matmul(-om,thetajt),thetajt.T)+np.identity(d)),np.matmul(Ut,Λt)) *thisphi
            t2sum/=num_samples
            Ut=(Ut-ht*np.matmul(Ut,Λt)-ht*t2sum)

            #update as per equation 8
            Ut, r = scipy.linalg.qr(Ut,mode='economic')

            if t%10==0: 
                #updating eigenvalues as per pdf 191
                om=omega(u0t,Ut,np.identity(p)) #as per pdf 165
                for i in range(p):
                    ui=Ut[:,i]
                    Λt[i][i]=np.matmul(np.matmul(ui.T,oactual),ui)-u0t**2

                om=omega(u0t,Ut,Λt) #as per pdf 165
                #suggested by mentor: Frobenius Norm
                fn=np.linalg.norm(oactual-om)
                estart=time.time()-start
                FNRT_By_u0[u0].append((estart,fn))
                if estart>30:
                    break
                logging.info('epoach {} p {} u0 {:.4f} Frobenius Norm {:.4f}'.format(t,p,u0,fn))
        if fn<bestfb:
            bestu0[p]=u0
            bestfb=fn

    fig = go.Figure()

    for u0 in [1,2,3,4,5]:
        XY=FNRT_By_u0[u0]
        y=np.log(np.array([y for x,y in XY]))
        x=np.array([x for x,y in XY])
        trace = go.Scatter(x=x, y=y, 
                           name='VI(u0={})'.format(u0))
        fig.add_trace(trace)

    fig.update_layout(
        legend=dict(
            x=0,
            y=-0.5,
            traceorder="normal",
            font=dict(
                family="sans-serif",
                size=12,
                color="black"
            ),
        )
    )

    fig.update_layout(title='Computational and Statistical Tradeoffs (p={} best u0={})'.format(p,bestu0[p]))
    fig.update_xaxes(title="Runtime(s)")
    fig.update_yaxes(title="log(Frobenius Norm) vs LT NUTS")
    fig.write_image('images/optimum_u0_p{}.png'.format(p))

bestu0={2: 4, 5: 4, 8: 4, 11: 3, 14: 3} #cached result

logging.info('capstone_vi_methods research done')

# Trade off Analysis for different dimensions p
FNRT_ByPValue=defaultdict(list)
for p in [2,5,8,11,14]:
    Λ = np.identity(p)
    U=np.eye(d,p)
    u0=bestu0[p]
    u0t,Ut,Λt=u0,U,Λ
    start=time.time()

    for t in range(10000):
        om=omega(u0t,Ut,Λt) #as per pdf 165

        num_samples=200+t*3
        samples,phis=getSamples(mu_star,om,num_samples) #Sample as per equation 6
        phis-=phis.mean()
        expphi=np.abs(phis).mean()        

        ht=0.01/(1+expphi)

        t2sum=0
        for thisphi,thetajt in zip(phis,samples):
            thetajt=(thetajt-mu_star).reshape((d,1))
            t2sum+=np.matmul((np.matmul(np.matmul(-om,thetajt),thetajt.T)+np.identity(d)),np.matmul(Ut,Λt)) *thisphi
        t2sum/=num_samples
        Ut=(Ut-ht*np.matmul(Ut,Λt)-ht*t2sum)

        #update as per equation 8
        Ut, r = scipy.linalg.qr(Ut,mode='economic')

        if t%10==0: 
            #updating eigenvalues as per pdf 191
            om=omega(u0t,Ut,np.identity(p)) #as per pdf 165
            for i in range(p):
                ui=Ut[:,i]
                Λt[i][i]=np.matmul(np.matmul(ui.T,oactual),ui)-u0t**2

            om=omega(u0t,Ut,Λt) #as per pdf 165
            #suggested by mentor: Frobenius Norm
            fn=np.linalg.norm(oactual-om)
            estart=time.time()-start
            FNRT_ByPValue[p].append((estart,fn))
            if estart>30:
                break
            logging.info('epoach {} avg phi {:.4f} Frobenius Norm {:.4f}'.format(t,expphi,fn))


import plotly.graph_objects as go

fig = go.Figure()

for p in [2,5,8,11,14]:
    XY=FNRT_ByPValue[p]
    y=np.log(np.array([y for x,y in XY]))
    x=np.array([x for x,y in XY])
    trace = go.Scatter(x=x, y=y, 
                       name='VI(p={})'.format(p))
    fig.add_trace(trace)

fig.update_layout(title='Computational and Statistical Tradeoffs')
fig.update_xaxes(title="Runtime(s)")
fig.update_yaxes(title="log(Frobenius Norm) vs LT NUTS")
fig.update_layout(
    legend=dict(
        x=0,
        y=-0.5,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
        ),
    )
)

fig.update_layout(title='Computational and Statistical Tradeoffs')
fig.update_xaxes(title="Runtime(s)")
fig.update_yaxes(title="log(Frobenius Norm) vs LT NUTS")
fig.write_image('images/VI_different_p.png')


png_renderer.width = 1000
png_renderer.height = 800

fig = go.Figure()

for p in [14]:
    XY=FNRT_ByPValue[p]
    y=np.log(np.array([y for x,y in XY]))
    x=np.array([x for x,y in XY])
    trace = go.Scatter(x=x, y=y, 
                       name='VI(p={})'.format(p))
    fig.add_trace(trace)
    trace = go.Scatter(x=x, y=y, mode='markers',
                       name='VI(p={})'.format(p))
    fig.add_trace(trace)

fig.update_layout(title='Computational and Statistical Tradeoffs')
fig.update_xaxes(title="Runtime(s)")
fig.update_yaxes(title="log(Frobenius Norm) vs LT NUTS")
fig.update_layout(
    legend=dict(
        x=1,
        y=1,
        traceorder="normal",
        font=dict(
            family="sans-serif",
            size=12,
            color="black"
        ),
    )
)

maxX=40
for res,l,col in [(ADVItimeToFN,'ADVI','blue'),(NUTStimeToFN,'NUTS','red')]:
    XY=sorted(zip(res.keys(),res.values()))
    X=np.array([x[0] for x in XY])
    y=np.array([np.log(x[1]) for x in XY])
    trace = go.Scatter(x=X[X<maxX], y=y[X<maxX],mode='markers',name=l)
    fig.add_trace(trace)
    #fit,param=np.polynomial.polynomial.Polynomial.fit(X,y, 6, full=True)
    #params,_=scipy.optimize.curve_fit(lambda t,a,b,c: a+b*np.log(t-c**2),  X,  y)
    params,_=scipy.optimize.curve_fit(lambda t,a,b,c: a+b*(1/np.clip(t+c,0.0001,None)),  X,  y)
    a=params[0]
    b=params[1]
    c=params[2]
    def fit(t):
        return a+b*(1/np.clip(t+c,0.0001,None))
    plotrange=np.arange(min(X),min(max(X)*1.1,maxX))
    trace = go.Scatter(x=plotrange, y=fit(plotrange),name=l)
    fig.add_trace(trace)

trace = go.Scatter(x=[sklearnruntime], y=[np.log(sklearnfn)],mode='markers',name='sklearn',
                  marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')))
fig.add_trace(trace)

    
fig.update_layout(title='Computational and Statistical Tradeoffs')
fig.update_xaxes(title="Runtime(s)")
fig.update_yaxes(title="log(Frobenius Norm) vs LT NUTS")
fig.write_image('images/FINAL_RESULT.png')

