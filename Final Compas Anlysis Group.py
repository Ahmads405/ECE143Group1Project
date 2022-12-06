#!/usr/bin/env python
# coding: utf-8

# # __An Analysis on The Reliability of The COMPAS Recidivism Algorithm__
# 
# <center>
#     <i>Ahmad Said, Yiyuan Cui, Berry Ma, Elavarthi Pragna, Muhammad Fadli Alim Arsani</i> 
# </center>

# In this project, we analyse the reliability of an algorithm used to predict the tendency of a convicted criminal to reoffend. In particular, __we argue that the COMPAS Recidivism Algorithm is biased towards certain races, specifically African-American__.
# 
# This notebook showcase our attempts on strengthening our claim using data analysis.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib import rcParams
import numpy as np
# figure size in inches
rcParams['figure.figsize'] = 25, 10

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("compas-scores-two-years.csv")


# ## Muhammad Fadli

# Here, we take a look at how `decile_score` (1-10, 10 being most likelike to recid) is related to `sex`, `age_cat` (age categories), and `race`. The x-axis is is the decile scores and the y-axis is the count of how many of the criminals correspond to that decile score.
# 
# We see that all 3 shows a trend, however, what we're interested in is the trend shown by the `race` feature since it is uncanny - African-American race is highly rated as being more likely to recid than other race. As the decile score increase, there is a significant difference in the count between the races.

# In[2]:


plt.subplot(121)
sns.histplot(data=df, palette=sns.color_palette("Set1", 2), shrink=.6, x='decile_score', hue='sex', multiple='dodge', discrete=True)
plt.xticks(list(range(1, 11)), list(range(1, 11)))

plt.subplot(122)
sns.histplot(data=df, palette=sns.color_palette("Set1", 3), shrink=.6, x='decile_score', hue='age_cat', multiple='dodge', discrete=True)
plt.xticks(list(range(1, 11)), list(range(1, 11)))

plt.show()

sns.histplot(data=df, palette=sns.color_palette("Set1", 6), shrink=.6, x='decile_score', hue='race', multiple='dodge', discrete=True)
plt.xticks(list(range(1, 11)), list(range(1, 11)))

plt.show()


# ## Yiyuan

# In[3]:


RACE_IN_FOCUS = ['African-American', 'Caucasian']
df_race_focused = df[df['race'].isin(RACE_IN_FOCUS)]
g = sns.FacetGrid(df_race_focused, col='race')
g.map(plt.hist, 'decile_score', rwidth=0.9,density = True)


# In[4]:


pd.crosstab(df_race_focused['decile_score'], df['race'], normalize = 'columns')


# In[5]:


recid_df = df.loc[df['two_year_recid']==1]
recid_df_race_focused = recid_df[recid_df['race'].isin(RACE_IN_FOCUS)]
g = sns.FacetGrid(recid_df_race_focused, col='race')
g.map(plt.hist, 'decile_score', rwidth=0.9, density = True)


# In[6]:


high_decile_recide = recid_df.loc[recid_df['decile_score'] > 6]
focused_high_decile = high_decile_recide[high_decile_recide['race'].isin(RACE_IN_FOCUS)]
g = sns.FacetGrid(focused_high_decile, col='race')
g.map(plt.hist, 'decile_score', rwidth=0.9,density = True)


# In[7]:


df2 = df


# In[8]:


df = df.fillna(0)
df = df.replace({'race' : { 'African-American' : 0, 'Asian' : 1, 'Caucasian' : 3 ,'Hispanic':4,'Native American':5,'Other':6}})
df = df.replace({'sex' : { 'Male' : 1, 'Female' : 0}})

corr_df = pd.DataFrame(columns=['r','p'])
for col in df:
    if pd.api.types.is_numeric_dtype(df[col]):
        r,p = stats.pearsonr(df.two_year_recid,df[col])
        corr_df.loc[col] = [round(r,3),round(p,3)]

cor_value = abs(corr_df.r).sort_values(ascending=False)
cor_value.head(20)


# In[9]:


g = sns.FacetGrid(df_race_focused, col='age_cat')
g.map(plt.hist, 'decile_score', rwidth=0.9)


# In[10]:


import numpy as np
df_race_focused["priors_count"] = np.where(df_race_focused["priors_count"] <= 5, 0, 1)
g = sns.FacetGrid(df_race_focused, col='priors_count')
g.map(plt.hist, 'decile_score', rwidth=0.9,density = True)


# ## Pragna Elavarthi

# In[11]:


list(df2.columns)


# Plotting a pie chart to show the distribution of data in category 'sex'

# In[12]:


k= df2['sex'].value_counts()


# In[13]:


k.plot.pie(subplots=True)


# ### Feature Selection

# Selecting important features for exploratory data analysis and discarding the rest by intuition

# In[14]:


data = df2[['sex','age','age_cat','race','juv_fel_count','decile_score','juv_misd_count','juv_other_count','priors_count','c_charge_degree','is_recid','r_charge_degree','is_violent_recid', 'type_of_assessment','decile_score.1','score_text', 'v_type_of_assessment', 'v_decile_score', 'v_score_text','priors_count.1','two_year_recid'  ]]


# In[15]:


data.head()


# In[16]:


list(data.columns)


# Finding correlation between age and recidivism in male

# One-hot encoding the category 'sex'

# In[17]:


h = pd.get_dummies(data['sex'])


# In[18]:


data = pd.concat([data, h], axis=1)


# In[19]:


data.head()


# In[20]:


temp = data[data.Male != 0]


# In[21]:


temp = temp[['age','is_recid' ]]


# In[22]:


temp.info()


# In[23]:


temp.head()


# In[24]:


temp.dropna()


# Comment: No missing values in these two columns

# In[25]:


temp['is_recid'].value_counts()


# In[26]:


temp['is_recid'].value_counts().value_counts().plot.pie(subplots=True)


# In[27]:


temp = pd.DataFrame(temp)


# In[28]:


temp.corr(method ='pearson')


# Pearson correlation coefficient(p-value) = -0.207637, this indicates a small negative correlation. That is, with increase in age tendency to relapse slightly decreases. However, a p-value of this low is considered insignificant which brings us to assume age and recidivism are not correlated in men.
# 
# Reasons for decrease in recidivism with age could be : 1) with age and experience comes maturity 2) as age increases, lifespan decreases...time to commit a crime decreases...

# Finding correlation between age category and recidivism in female

# In[29]:


temp = data[data.Female != 0]


# In[30]:


temp = temp[['age','is_recid' ]]


# In[31]:


temp.info()


# In[32]:


temp['is_recid'].value_counts().value_counts().plot.pie(subplots=True)


# In[33]:


temp = pd.DataFrame(temp)


# In[34]:


temp.corr(method ='pearson')


# Pearson correlation coefficient(p-value) = -0.155151, this indicates a small negative correlation, the correlation weaker than in men. That is, with increase in age tendency to relapse slightly decreases. However, a p-value of this low is considered insignificant which brings us to assume age and recidivism are not correlated in women. Reasons could be the same as above

# Plotting number of total cases by sex and age category
# 
# 

# In[35]:


data['age_cat'].value_counts()


# In[36]:


plt.figure(figsize=(8,5))
sns.countplot(data=data,x='sex', hue="age_cat",palette="Blues_r")
plt.title("Number of cases by sex and age_cat")
plt.xlabel("")
plt.show(block=False)


# In[37]:


temp = temp = data[data.is_recid != 0]


# Plotting number of recid cases by sex and age category

# In[38]:


temp['age_cat'].value_counts()


# In[39]:


plt.figure(figsize=(8,5))
sns.countplot(data=temp,x="sex",hue="age_cat",palette="Blues_r")
plt.title("Number of cases by sex and age_cat")
plt.xlabel("")
plt.show(block=False)


# c_charge_degree

# In[40]:


data['c_charge_degree'].value_counts()


# In[41]:


temp = data[['c_charge_degree','is_recid','r_charge_degree' ]]


# In[42]:


plt.figure(figsize=(4,5))
sns.countplot(data=temp,x="c_charge_degree",palette="Blues_r")
plt.title("Number of cases by type of charge")
plt.xlabel("")
plt.show(block=False)


# In[43]:


temp['r_charge_degree'].head()


# In[44]:


temp.dropna()


# In[45]:


temp['r_charge_degree'].value_counts()


# In[46]:


plt.figure(figsize=(8,5))
sns.countplot(data=temp,x="r_charge_degree",hue="c_charge_degree",palette="Blues_r")
plt.title("Number of cases by type of recid charge and initial crime charge")
plt.xlabel("")
plt.show(block=False)


# ## Berry

# In[47]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
df = pd.read_csv("compas-scores-two-years.csv")
rcParams['figure.figsize'] = 5, 5


# We try to use the dataset itself to do some prediction. By adding a new column, we can translate the risk score (decile_score) into a binary label.\
# A score of 5 or higher (Medium or High risk) suggests that one is more likely to be a recividist, and a score of 4 or lower (Low risk) means one is unlikely to re-offend.

# In[48]:


df['is_med_or_high_risk']  = (df['decile_score']>=5).astype(int)


# To evaluate the prediction, we will compare the predictions to the “truth”

# In[49]:


# classification results
np.mean(df['is_med_or_high_risk']==df['two_year_recid'])
np.mean(df['two_year_recid'])


# This looks problematic.
# There are two kinds of errors in the prediction.
# 
# + false positives (predicted as medium/high risk but does not re-offend)
# + false negatives (predicted as low risk, but does re-offend)
# 
# By creating a confusion matrix, we can pull them out separately, to see different types of errors.

# In[50]:


cm = pd.crosstab(df['is_med_or_high_risk'], df['two_year_recid'], 
                               rownames=['Predicted'], colnames=['Actual'])
p = plt.figure(figsize=(5,5));
p = sns.heatmap(cm, annot=True, fmt="d", cbar=False)


# In[51]:


[[tn , fp],[fn , tp]]  = confusion_matrix(df['two_year_recid'], df['is_med_or_high_risk'])
print("True negatives:  ", tn)
print("False positives: ", fp)
print("False negatives: ", fn)
print("True positives:  ", tp)


# normalize by row

# In[52]:


cm = pd.crosstab(df['is_med_or_high_risk'], df['two_year_recid'], 
                               rownames=['Predicted'], colnames=['Actual'], normalize='index')
p = plt.figure(figsize=(5,5));
p = sns.heatmap(cm, annot=True, fmt=".2f", cbar=False)


# normalize by column

# In[53]:


cm = pd.crosstab(df['is_med_or_high_risk'], df['two_year_recid'], 
                               rownames=['Predicted'], colnames=['Actual'], normalize='columns')
p = plt.figure(figsize=(5,5));
p = sns.heatmap(cm, annot=True, fmt=".2f", cbar=False)


# We see that a defendant has a similar likelihood of being wrongly labeled a likely recidivist and of being wrongly labeled as unlikely to re-offend.

# In[54]:


fpr = fp/(fp+tn)
fnr  = fn/(fn+tp)
 
 
print("False positive rate (overall): ", fpr)
print("False negative rate (overall): ", fnr)


# We could also explore the relation between risk score and recidivism.

# In[55]:


d = df.groupby('decile_score').agg({'two_year_recid': 'mean'})
sns.scatterplot(data=d);
plt.ylim(0,1);
plt.ylabel('Recidivism rate');


# Receiver Operating Characteristic curve(ROC curve) and Area Under the Curve(AUC) are often used to evaluate a binary classifier.\
# The ROC curve is created by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings.\
# The Area Under the Curve (AUC) is the measure of the ability of a classifier to distinguish between classes and is used as a summary of the ROC curve.

# In[56]:


fpr, tpr, thresholds = roc_curve(df['two_year_recid'], df['decile_score'])
sns.scatterplot(x=fpr, y=tpr, );
sns.lineplot(x=fpr, y=tpr);
plt.ylabel("TPR");
plt.xlabel("FPR");


# In[57]:


auc = roc_auc_score(df['two_year_recid'], df['decile_score'])
auc


# We find out how frequently each race is represented in the data

# In[58]:


df['race'].value_counts()


# We focus specifically on African-American or Caucasian.\
# We calculate the prediction accuracy like we did above to these two groups.

# In[59]:


df = df[df.race.isin(["African-American","Caucasian"])]
# compare accuracy
(df['two_year_recid']==df['is_med_or_high_risk']).astype(int).groupby(df['race']).mean()


# Now let’s see whether a defendant classified as medium/high risk has the same probability of recidivism for the two groups.\
# In other words, we calculate the Positive Predictive Value (PPV).

# In[60]:


#PPV
df[df['is_med_or_high_risk']==1]['two_year_recid'].groupby(df['race']).mean()


# Now we check whether a defendant with a given score has the same probability of recidivism for the two groups.

# In[61]:


d = pd.DataFrame(df.groupby(['decile_score','race']).agg({'two_year_recid': 'mean'}))
d = d.reset_index()
im = sns.scatterplot(data=d, x='decile_score', y='two_year_recid', hue='race');
im.set(ylim=(0,1));


# Now we look at the frequency with which defendants of each race are assigned each COMPAS score

# In[62]:


g = sns.FacetGrid(df, col="race", margin_titles=True);
g.map(plt.hist, "decile_score", bins=10);
df.groupby('race').agg({'two_year_recid': 'mean',  
                        'is_med_or_high_risk': 'mean', 
                        'decile_score': 'mean'})


# We see that Caucasians are more likely to be assigned a low risk score.\
# Now we try to fix this with different threshold.

# In[63]:


black_threshold  = 5
df_black = df[df['race']=="African-American"].copy()
df_black['is_med_or_high_risk'] = (df_black['decile_score']>=black_threshold).astype(int)
[[tn , fp],[fn , tp]]  = confusion_matrix(df_black['two_year_recid'], df_black['is_med_or_high_risk'])
print("False positive rate (Black)      : ", fp/(fp+tn))
print("False negative rate (Black)      : ", fn/(fn+tp))
print("Positive predictive value (Black): ", tp/(tp+fp))
print("Negative predictive value (Black): ", tn/(tn+fn))


# In[64]:


white_threshold  = 5
df_white = df[df['race']=="Caucasian"].copy()
df_white['is_med_or_high_risk'] = (df_white['decile_score']>=white_threshold).astype(int)
[[tn , fp],[fn , tp]]  = confusion_matrix(df_white['two_year_recid'], df_white['is_med_or_high_risk'])
print("False positive rate (white)      : ", fp/(fp+tn))
print("False negative rate (white)      : ", fn/(fn+tp))
print("Positive predictive value (white): ", tp/(tp+fp))
print("Negative predictive value (white): ", tn/(tn+fn))


# Now we make a relatively equal FPR and FNR for both groups

# # Ahmad

# In[65]:


from scipy.stats import ttest_ind
from scipy.stats import ttest_rel
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib import rcParams
from sklearn.metrics import roc_curve, roc_auc_score
# figure size in inches
rcParams['figure.figsize'] = 25, 10

df = pd.read_csv("compas-scores-two-years.csv")


# In[66]:


group1 = df[df['race']=='Caucasian']
group2 = df[df['race']=='African-American']
ttest_ind(group1['two_year_recid'], group2['two_year_recid'],equal_var=False)


# In[67]:


RACE_IN_FOCUS = ['African-American', 'Caucasian']

recid_df = df.loc[df['two_year_recid']==0]
recid_df_race_focused = recid_df[recid_df['race'].isin(RACE_IN_FOCUS)]
g = sns.FacetGrid(recid_df_race_focused, col='race')
g.map(plt.hist, 'decile_score', rwidth=0.9, density = True)


# In[68]:


RACE_IN_FOCUS = ['African-American', 'Caucasian']
df_race_focused = df[df['race'].isin(RACE_IN_FOCUS)]
g = sns.FacetGrid(df_race_focused, col='race')
g.map(plt.hist, 'two_year_recid', rwidth=0.9,density = True)


# In[69]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
black_df=df.loc[df['race']=='African-American']
white_df=df.loc[df['race']=='White']
fpr, tpr, thresholds = roc_curve(black_df['two_year_recid'], black_df['decile_score'])
sns.scatterplot(x=fpr, y=tpr, );
sns.lineplot(x=fpr, y=tpr);
plt.ylabel("TPR");
plt.xlabel("FPR");
plt.title('African-American')


# In[70]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
black_df=df.loc[df['race']=='African-American']
white_df=df.loc[df['race']=='Caucasian']
fpr, tpr, thresholds = roc_curve(white_df['two_year_recid'], white_df['decile_score'])
sns.scatterplot(x=fpr, y=tpr, );
sns.lineplot(x=fpr, y=tpr);
plt.ylabel("TPR");
plt.xlabel("FPR");
plt.title('White')


# In[71]:


no_recid_df=df.loc[df['two_year_recid']==0]
recid_df=df.loc[df['two_year_recid']==1]

print(no_recid_df['race'].value_counts(),'\n')
print(recid_df['race'].value_counts())


# In[ ]:




