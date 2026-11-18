import os,joblib,pandas as pd,streamlit as st,plotly.express as px
from ml_pipeline import engineer
st.set_page_config(page_title="ScholarAI",page_icon="🎓",layout="wide")
st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;600;700&family=Manrope:wght@600;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:linear-gradient(135deg,#08111f,#101b2d 60%,#111827);color:#edf3ff}
[data-testid="stSidebar"]{background:#0b1423;border-right:1px solid #26364e}
.block-container{padding-top:2rem;max-width:1450px}
.hero{padding:27px 30px;border:1px solid #2b4261;border-radius:22px;background:linear-gradient(120deg,#18304e,#14223a);margin-bottom:22px}
.hero h1{font-family:Manrope,sans-serif;font-size:2.1rem;font-weight:800;margin:0;color:#f5f8ff}
.hero p{color:#a9bdd8;margin:.4rem 0 0}
.stTabs [data-baseweb="tab"]{background:#172943;border-radius:10px 10px 0 0;color:#d7e5f7}
div[data-testid="stMetric"]{background:#172943;border:1px solid #2b4261;padding:14px;border-radius:14px}
.stButton>button{border-radius:11px;font-weight:700}
</style>""",unsafe_allow_html=True)
st.markdown('<div class="hero"><h1>🎓 ScholarAI Intelligence</h1><p>Advanced scholarship analytics · Classification · Regression · Explainable ML</p></div>',unsafe_allow_html=True)
if not all(os.path.exists(p) for p in ["models/classifier.joblib","models/regressor.joblib"]):
 st.error("Models missing. Run python train_models.py first."); st.stop()
@st.cache_resource
def load(): return joblib.load("models/classifier.joblib"),joblib.load("models/regressor.joblib")
cb,rb=load(); data=pd.read_csv("scholarship_dataset.csv")
with st.sidebar:
 st.header("Student assessment")
 with st.form("student"):
  income=st.number_input("Annual family income (₹)",0,10000000,250000,10000)
  size=st.number_input("Family members",1,20,4)
  category=st.selectbox("Category",["General","OBC","SC","ST"])
  residence=st.selectbox("Residence",["Urban","Rural"])
  tenth=st.slider("Class 10 (%)",0.,100.,78.)
  twelfth=st.slider("Class 12 (%)",0.,100.,75.)
  cgpa=st.slider("CGPA",0.,10.,7.5)
  attendance=st.slider("Attendance (%)",0.,100.,85.)
  year=st.selectbox("Course year",[1,2,3,4]); backlogs=st.selectbox("Backlogs",[0,1,2,3])
  go=st.form_submit_button("Run AI assessment",use_container_width=True)
t1,t2,t3,t4,t5=st.tabs(["Overview","Prediction","Model laboratory","Explainability","Project notes"])
with t1:
 a,b,c,d=st.columns(4); a.metric("Synthetic records",f"{len(data):,}"); b.metric("Positive labels",f"{data.eligible.sum():,}")
 c.metric("Raw inputs","10"); d.metric("Engineered features","3")
 x,y=st.columns(2)
 with x: st.plotly_chart(px.histogram(data,x="annual_family_income",color="eligible",nbins=28,title="Income distribution by synthetic label",template="plotly_dark"),use_container_width=True)
 with y: st.plotly_chart(px.histogram(data,x="category",color="category",title="Records by category",template="plotly_dark"),use_container_width=True)
with t2:
 st.subheader("Individual prediction")
 if go:
  row=pd.DataFrame([dict(annual_family_income=income,family_size=size,category=category,tenth_percentage=tenth,twelfth_percentage=twelfth,cgpa=cgpa,attendance=attendance,course_year=year,backlogs=backlogs,residence=residence)])
  feat=engineer(row); prob=float(cb["model"].predict_proba(feat)[0,1]); pred=int(cb["model"].predict(feat)[0])
  left,right=st.columns(2)
  with left:
   st.success("Likely eligible (model output)") if pred else st.warning("Not eligible")
   st.metric("Model probability for eligible class",f"{prob:.1%}"); st.progress(prob)
   st.caption("Classifier: "+cb["name"])
  with right:
   if pred: st.metric("Estimated synthetic amount",f"₹{max(0,float(rb['model'].predict(feat)[0])):,.0f}")
   else: st.metric("Estimated amount","Not estimated")
   st.caption("Regressor: "+rb["name"])
  z1,z2,z3=st.columns(3); z1.metric("Income/member",f"₹{feat.iloc[0].income_per_member:,.0f}"); z2.metric("Academic score",f"{feat.iloc[0].academic_score:.1f}"); z3.metric("Financial need index",f"{feat.iloc[0].financial_need_index:.1f}/100")
  st.info("Demo prediction only. It is not an official scholarship decision or guarantee.")
 else: st.info("Enter student details in the sidebar and click Run AI assessment.")
with t3:
 st.subheader("Classification model comparison")
 cp="reports/classification_comparison.csv"; rp="reports/regression_comparison.csv"
 if os.path.exists(cp):
  cdf=pd.read_csv(cp); st.dataframe(cdf.round(3),use_container_width=True,hide_index=True)
  st.plotly_chart(px.bar(cdf,x="model",y=["accuracy","precision","recall","f1","roc_auc"],barmode="group",template="plotly_dark"),use_container_width=True)
 if os.path.exists(rp):
  rdf=pd.read_csv(rp); st.subheader("Regression model comparison"); st.dataframe(rdf.round(2),use_container_width=True,hide_index=True)
  st.plotly_chart(px.bar(rdf,x="model",y="mae",color="model",template="plotly_dark",title="MAE — lower is better"),use_container_width=True)
 cm="reports/confusion_matrix.csv"
 if os.path.exists(cm):
  st.subheader("Confusion matrix"); st.plotly_chart(px.imshow(pd.read_csv(cm,index_col=0),text_auto=True,template="plotly_dark",color_continuous_scale="Blues"),use_container_width=True)
with t4:
 st.subheader("Permutation feature importance")
 fp="reports/feature_importance.csv"; sp="reports/subgroup_review.csv"
 if os.path.exists(fp):
  imp=pd.read_csv(fp); st.plotly_chart(px.bar(imp.head(12).sort_values("importance_mean"),x="importance_mean",y="feature",orientation="h",error_x="importance_std",template="plotly_dark"),use_container_width=True); st.dataframe(imp,hide_index=True,use_container_width=True)
 if os.path.exists(sp):
  st.subheader("Descriptive subgroup error review"); st.dataframe(pd.read_csv(sp),hide_index=True,use_container_width=True)
  st.caption("Diagnostic group rates only; not a fairness certification.")
with t5:
 st.markdown("""**Classification:** Logistic Regression, Decision Tree, Random Forest, Gradient Boosting and tuned Random Forest.
**Regression:** Linear Regression, Decision Tree, Random Forest and Gradient Boosting.
**Training:** 5-fold cross-validation, held-out test set, GridSearchCV, model comparison.
**Feature engineering:** income/member, weighted academic score, financial-need index.
**Preprocessing:** imputation, scaling, one-hot encoding in Scikit-learn pipelines.""")
 st.error("The included dataset is synthetic. Its labels and scholarship amounts are generated from illustrative formulas, not real outcomes or official scheme rules. Do not use for real applicant decisions.")
