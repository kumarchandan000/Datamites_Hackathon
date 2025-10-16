#!/usr/bin/env python
# coding: utf-8

# In[117]:


import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder,OrdinalEncoder
import warnings
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

import pickle
import warnings
warnings.filterwarnings('ignore')


# In[75]:


df=pd.read_csv("female_foeticide_risk.csv")


# In[76]:


df.head(10)


# In[40]:


df.shape


# In[8]:


df.describe()


# In[7]:


df.isnull().sum()


# In[66]:


# Imputing the null values :
df.loc[df['literacy_rate_female'].isnull(),'literacy_rate_female']=df["literacy_rate_female"].median() # We have imputed the null values with the median


# In[10]:


df.literacy_rate_female.isnull().sum()


# In[11]:


df.loc[df['literacy_rate_male'].isnull(),'literacy_rate_male']=df["literacy_rate_male"].median()


# In[12]:


df.literacy_rate_male.isnull().sum()


# In[13]:


df.loc[df['sex_ratio'].isnull(),'sex_ratio']=df["sex_ratio"].median()


# In[14]:


df.sex_ratio.isnull().sum()


# In[27]:


df.loc[df['avg_household_income'].isnull(),'avg_household_income']=df["avg_household_income"].mean()


# In[28]:


df.avg_household_income.isnull().sum()


# In[20]:


df.loc[df['district_name'].isnull(),'district_name']='0'


# In[21]:


df.district_name.isnull().sum()


# In[22]:


df.loc[df['state'].isnull(),'state']='0'


# In[23]:


df.state.isnull().sum()


# In[29]:


df.loc[df['poverty_index'].isnull(),'poverty_index']=df["poverty_index"].mean()


# In[30]:


df.poverty_index.isnull().sum()


# In[41]:


df['education_expenditure_per_capita'].fillna(0.00, inplace=True)


# In[42]:


df.education_expenditure_per_capita.isnull().sum()


# In[43]:


df.loc[df['female_infant_mortality_rate'].isnull(),'female_infant_mortality_rate']=df["female_infant_mortality_rate"].median()


# In[44]:


df.female_infant_mortality_rate.isnull().sum()


# In[49]:


df['access_to_health_facilities'].fillna(0.00, inplace=True)


# In[50]:


df.access_to_health_facilities.isnull().sum()


# In[51]:


df.loc[df['employment_rate_female'].isnull(),'employment_rate_female']=df["employment_rate_female"].median()


# In[52]:


df.employment_rate_female.isnull().sum()


# In[53]:


df['social_awareness_programs'].fillna(0.00, inplace=True)


# In[54]:


df.isnull().sum()


# In[55]:


df.columns


# In[56]:


df.info()


# In[61]:


df.risk_level.value_counts()


# ### Data Exploration

# In[57]:


sns.boxplot(x=df.employment_rate_female ,orient='h')


# In[63]:


sns.countplot(x='risk_level', data=df, palette='coolwarm')
plt.title("Risk Level Distribution")
plt.show()

print(df['risk_level'].value_counts(normalize=True))


# In[64]:


num_cols = [
    'literacy_rate_female','literacy_rate_male','sex_ratio','avg_household_income',
    'poverty_index','education_expenditure_per_capita','female_infant_mortality_rate',
    'access_to_health_facilities','employment_rate_female','social_awareness_programs'
]

# Histograms
df[num_cols].hist(figsize=(15,12), bins=20, color='#1f77b4')
plt.suptitle("Numeric Feature Distributions", fontsize=16)
plt.show()

# Boxplots (to see outliers)
plt.figure(figsize=(14,8))
sns.boxplot(data=df[num_cols])
plt.xticks(rotation=45)
plt.title("Boxplots — Checking for Outliers")
plt.show()


# In[65]:


plt.figure(figsize=(10,8))
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()


# In[67]:


state_risk = df.groupby('state')['risk_level'].value_counts(normalize=True).unstack().fillna(0)
state_risk.plot(kind='bar', stacked=True, figsize=(12,6), colormap='RdYlBu')
plt.title("Risk Level Distribution by State")
plt.ylabel("Proportion")
plt.show()


# In[68]:


features_to_plot = ['literacy_rate_female','sex_ratio','poverty_index','employment_rate_female','female_infant_mortality_rate']

for feat in features_to_plot:
    plt.figure(figsize=(6,4))
    sns.boxplot(x='risk_level', y=feat, data=df, palette='coolwarm')
    plt.title(f"{feat} vs Risk Level")
    plt.show()


# # Feature Engineering

# In[105]:


df['literacy_gap'] = df['literacy_rate_male'] - df['literacy_rate_female']
df['female_lit_ratio'] = df['literacy_rate_female'] / (df['literacy_rate_male'] + 1e-6)
df['log_income'] = np.log1p(df['avg_household_income'])

# Map target to integers
target_map = {'Low':0, 'Medium':1, 'High':2}
df['y'] = df['risk_level'].map(target_map)


# In[106]:


# Feature Lists

num_feats = [
    'literacy_rate_female','literacy_rate_male','sex_ratio','log_income',
    'poverty_index','education_expenditure_per_capita','female_infant_mortality_rate',
    'access_to_health_facilities','employment_rate_female','social_awareness_programs',
    'literacy_gap','female_lit_ratio'
]
cat_feats = ['state']

X = df[num_feats + cat_feats]
y = df['y']


# In[107]:


# Preprocessing Pipelines
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
# Numeric: median impute + standard scaling
num_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='median')),
    ('scale', StandardScaler())
])

# Categorical: most frequent impute + one-hot encode
cat_pipeline = Pipeline([
    ('impute', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))  
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_feats),
    ('cat', cat_pipeline, cat_feats)
])


# In[108]:


# Modeling Pipeline (SMOTE + XGBoost)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

clf = XGBClassifier(
    objective='multi:softprob',
    eval_metric='mlogloss',
    use_label_encoder=False,
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    random_state=42,
    n_jobs=-1
)

pipeline = ImbPipeline(steps=[
    ('preproc', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('clf', clf)
])


# In[109]:


# Train/Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# In[110]:


# Fit Model
pipeline.fit(X_train, y_train)


# In[111]:


# cross-validated predictions
from sklearn.model_selection import StratifiedKFold, cross_val_predict
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(pipeline, X, y, cv=skf, method='predict', n_jobs=1)


# In[112]:


# Predictions & Evaluation of the model
from sklearn.metrics import f1_score, classification_report, confusion_matrix

y_pred = pipeline.predict(X_test)

print("Macro F1 Score:", f1_score(y_test, y_pred, average='macro'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))


# In[113]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1_macro')
print("5-fold CV Macro F1 Score:", scores)
print("Average Macro F1 Score:", scores.mean())


# In[114]:


# Examples of  new data
X_new = pd.DataFrame([{
    'literacy_rate_female': 75.0,
    'literacy_rate_male': 85.0,
    'sex_ratio': 950,
    'log_income': 2.5,
    'poverty_index': 0.2,
    'education_expenditure_per_capita': 500,
    'female_infant_mortality_rate': 35,
    'access_to_health_facilities': 0.8,
    'employment_rate_female': 0.6,
    'social_awareness_programs': 10,
    'literacy_gap': 10,
    'female_lit_ratio': 0.88,
    'state': 'Karnataka'
}])


# In[115]:


y_new_pred = pipeline.predict(X_new)
print("Predicted class:", y_new_pred[0])


# ## Here we can see the with the new data values the predicted class is 0 which means "low risk".

# In[116]:


import joblib

# For Saving the pickel file.
joblib.dump(pipeline, "final_model.pkl")


# In[ ]:




