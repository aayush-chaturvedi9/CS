import os,json,joblib,numpy as np,pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split,StratifiedKFold,KFold,cross_val_score,GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,classification_report,mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,RandomForestRegressor,GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from ml_pipeline import RAW,FEATURES,engineer,preprocessor
os.makedirs("models",exist_ok=True); os.makedirs("reports",exist_ok=True)
df=pd.read_csv("scholarship_dataset.csv"); X=engineer(df); y=df.eligible.astype(int)
Xt,Xv,yt,yv=train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
cv=StratifiedKFold(5,shuffle=True,random_state=42)
cls={"Logistic Regression":LogisticRegression(max_iter=1500,class_weight="balanced"),
"Decision Tree":DecisionTreeClassifier(max_depth=7,min_samples_leaf=8,class_weight="balanced",random_state=42),
"Random Forest":RandomForestClassifier(n_estimators=180,min_samples_leaf=3,class_weight="balanced",random_state=42,n_jobs=-1),
"Gradient Boosting":GradientBoostingClassifier(n_estimators=120,learning_rate=.05,max_depth=2,random_state=42)}
pipes={}; rows=[]
for name,model in cls.items():
 p=Pipeline([("prep",preprocessor()),("model",model)])
 scores=cross_val_score(p,Xt,yt,cv=cv,scoring="f1",n_jobs=-1); p.fit(Xt,yt); pred=p.predict(Xv); prob=p.predict_proba(Xv)[:,1]
 pipes[name]=p; rows.append(dict(model=name,cv_f1=scores.mean(),accuracy=accuracy_score(yv,pred),precision=precision_score(yv,pred,zero_division=0),recall=recall_score(yv,pred,zero_division=0),f1=f1_score(yv,pred,zero_division=0),roc_auc=roc_auc_score(yv,prob)))
grid=GridSearchCV(Pipeline([("prep",preprocessor()),("model",RandomForestClassifier(class_weight="balanced",random_state=42,n_jobs=-1))]),
 {"model__n_estimators":[140,220],"model__max_depth":[None,12],"model__min_samples_leaf":[2,4]},scoring="f1",cv=cv,n_jobs=-1)
grid.fit(Xt,yt); pipes["Tuned Random Forest"]=grid.best_estimator_; pred=grid.predict(Xv); prob=grid.predict_proba(Xv)[:,1]
rows.append(dict(model="Tuned Random Forest",cv_f1=grid.best_score_,accuracy=accuracy_score(yv,pred),precision=precision_score(yv,pred,zero_division=0),recall=recall_score(yv,pred,zero_division=0),f1=f1_score(yv,pred,zero_division=0),roc_auc=roc_auc_score(yv,prob)))
cr=pd.DataFrame(rows).sort_values("f1",ascending=False); bestc=pipes[cr.iloc[0].model]
joblib.dump({"model":bestc,"name":cr.iloc[0].model},"models/classifier.joblib"); cr.to_csv("reports/classification_comparison.csv",index=False)
bp=bestc.predict(Xv); bprob=bestc.predict_proba(Xv)[:,1]
pd.DataFrame(confusion_matrix(yv,bp),index=["Actual 0","Actual 1"],columns=["Predicted 0","Predicted 1"]).to_csv("reports/confusion_matrix.csv")
open("reports/classification_report.txt","w").write(classification_report(yv,bp,zero_division=0))
perm=permutation_importance(bestc,Xv,yv,n_repeats=6,random_state=42,scoring="f1",n_jobs=-1)
pd.DataFrame({"feature":FEATURES,"importance_mean":perm.importances_mean,"importance_std":perm.importances_std}).sort_values("importance_mean",ascending=False).to_csv("reports/feature_importance.csv",index=False)
# Regression trained only on rows marked eligible in this synthetic dataset.
mask=df.eligible.eq(1); Xr=engineer(df.loc[mask]); yr=df.loc[mask,"scholarship_amount"].astype(float)
ra,rb,ya,yb=train_test_split(Xr,yr,test_size=.2,random_state=42); kf=KFold(5,shuffle=True,random_state=42)
regs={"Linear Regression":LinearRegression(),"Decision Tree":DecisionTreeRegressor(max_depth=8,min_samples_leaf=5,random_state=42),
"Random Forest":RandomForestRegressor(n_estimators=180,min_samples_leaf=3,random_state=42,n_jobs=-1),
"Gradient Boosting":GradientBoostingRegressor(n_estimators=120,learning_rate=.05,max_depth=2,random_state=42)}
rpipes={}; rr=[]
for name,m in regs.items():
 p=Pipeline([("prep",preprocessor()),("model",m)]); cvmae=-cross_val_score(p,ra,ya,cv=kf,scoring="neg_mean_absolute_error",n_jobs=-1)
 p.fit(ra,ya); q=p.predict(rb); rpipes[name]=p
 rr.append(dict(model=name,cv_mae=cvmae.mean(),mae=mean_absolute_error(yb,q),rmse=np.sqrt(mean_squared_error(yb,q)),r2=r2_score(yb,q)))
rrdf=pd.DataFrame(rr).sort_values("mae"); bestr=rpipes[rrdf.iloc[0].model]
joblib.dump({"model":bestr,"name":rrdf.iloc[0].model},"models/regressor.joblib"); rrdf.to_csv("reports/regression_comparison.csv",index=False)
sub=df.loc[Xv.index,["category"]].copy(); sub["actual"]=yv; sub["predicted"]=bp
out=[]
for group,g in sub.groupby("category"):
 out.append(dict(category=group,count=len(g),selection_rate=g.predicted.mean(),false_negative_rate=((g.actual==1)&(g.predicted==0)).sum()/max(1,(g.actual==1).sum())))
pd.DataFrame(out).to_csv("reports/subgroup_review.csv",index=False)
json.dump({"best_classifier":cr.iloc[0].model,"best_regressor":rrdf.iloc[0].model,"grid_search_best_params":grid.best_params_,"warning":"Synthetic educational dataset only."},open("reports/training_summary.json","w"),indent=2)
print("Classification comparison:\\n",cr.to_string(index=False)); print("\\nRegression comparison:\\n",rrdf.to_string(index=False))
print("\\nModels and reports saved.")
