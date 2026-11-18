import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
RAW=["annual_family_income","family_size","category","tenth_percentage","twelfth_percentage","cgpa","attendance","course_year","backlogs","residence"]
NUM=["annual_family_income","family_size","tenth_percentage","twelfth_percentage","cgpa","attendance","course_year","backlogs","income_per_member","academic_score","financial_need_index"]
CAT=["category","residence"]; FEATURES=NUM+CAT
def engineer(frame):
    x=frame[RAW].copy()
    x["income_per_member"]=x.annual_family_income/x.family_size.replace(0,np.nan)
    x["academic_score"]=.18*x.tenth_percentage+.22*x.twelfth_percentage+5.2*x.cgpa
    x["financial_need_index"]=(100-x.income_per_member/14000).clip(0,100)
    return x[FEATURES]
def preprocessor():
    n=Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())])
    c=Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num",n,NUM),("cat",c,CAT)])
