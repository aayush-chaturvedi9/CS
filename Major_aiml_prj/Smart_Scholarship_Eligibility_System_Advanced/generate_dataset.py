import numpy as np, pandas as pd
r=np.random.default_rng(2026); n=3000
income=r.integers(40000,1500001,n); size=r.integers(1,9,n)
cat=r.choice(["General","OBC","SC","ST"],n,p=[.46,.27,.17,.10])
tenth=np.clip(r.normal(76,11,n),35,100); twelfth=np.clip(r.normal(74,12,n),35,100)
cgpa=np.clip(r.normal(7.25,1.15,n),4,10); att=np.clip(r.normal(83,10,n),40,100)
year=r.integers(1,5,n); back=r.choice([0,1,2,3],n,p=[.72,.17,.08,.03])
res=r.choice(["Urban","Rural"],n,p=[.58,.42])
ipm=income/size; acad=.18*tenth+.22*twelfth+5.2*cgpa
need=np.clip(100-ipm/14000,0,100)
bonus=pd.Series(cat).map({"General":0,"OBC":3,"SC":6,"ST":7}).to_numpy()
logit=2.25-income/510000+(cgpa-6)*.48+(att-70)*.022+(acad-70)*.03+bonus*.075+(res=="Rural")*.16-back*.38+r.normal(0,.9,n)
eligible=(1/(1+np.exp(-logit))>.54).astype(int)
amount=np.where(eligible,np.clip(12000+need*390+(acad-60)*300+bonus*420-back*900+r.normal(0,4200,n),8000,120000),0)
df=pd.DataFrame(dict(annual_family_income=income,family_size=size,category=cat,tenth_percentage=tenth,twelfth_percentage=twelfth,cgpa=cgpa,attendance=att,course_year=year,backlogs=back,residence=res,eligible=eligible,scholarship_amount=amount.astype(int)))
for c in ["attendance","twelfth_percentage","cgpa"]: df.loc[r.random(n)<.012,c]=np.nan
df.to_csv("scholarship_dataset.csv",index=False)
print(f"Generated {n} synthetic demonstration records.")
