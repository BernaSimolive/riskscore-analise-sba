import numpy as np, pandas as pd, json
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
USE=["NAICS","ApprovalFY","Term","NoEmp","NewExist","FranchiseCode","UrbanRural","RevLineCr","LowDoc","MIS_Status","GrAppv"]
df=pd.read_csv("SBAnational.csv",dtype=str,encoding="latin-1",usecols=USE)
df["GrAppv"]=pd.to_numeric(df["GrAppv"].str.replace(r"[$,\s]","",regex=True),errors="coerce")
for c in ["NoEmp","Term","NewExist","UrbanRural","FranchiseCode"]: df[c]=pd.to_numeric(df[c],errors="coerce")
df["FY"]=pd.to_numeric(df["ApprovalFY"].str.replace("A","",regex=False),errors="coerce")
df=df[df["MIS_Status"].isin(["P I F","CHGOFF"])&df["NewExist"].isin([1,2])&(df.FY>=1990)&(df.FY<=2010)&(df.Term>0)&(df.GrAppv>0)].copy()
df["y"]=(df.MIS_Status=="CHGOFF").astype(int); df["novo"]=(df.NewExist==2).astype(int)
df["franq"]=(df.FranchiseCode>1).astype(int); df["lowdoc"]=(df.LowDoc=="Y").astype(int); df["rev"]=(df.RevLineCr=="Y").astype(int)
df["setor"]=pd.to_numeric(df.NAICS.str[:2],errors="coerce").fillna(0)
df["urb"]=df.UrbanRural.fillna(0)
sets={"so_empresa (porte,novo,setor,urbano,franquia)":["NoEmp","novo","setor","urb","franq"],
      "empresa+prazo":["NoEmp","novo","setor","urb","franq","Term"],
      "empresa+valor":["NoEmp","novo","setor","urb","franq","GrAppv"],
      "empresa+contrato (prazo,valor,lowdoc,revolvente)":["NoEmp","novo","setor","urb","franq","Term","GrAppv","lowdoc","rev"]}
tr=df[df.FY<=2004]; te=df[df.FY>=2005]
rng=np.random.default_rng(1); sub=rng.choice(len(tr),150000,replace=False)
out={}
for k,cols in sets.items():
    gb=HistGradientBoostingClassifier(random_state=1,max_iter=80,categorical_features=[i for i,c in enumerate(cols) if c=="setor"] or None)
    gb.fit(tr[cols].iloc[sub],tr.y.iloc[sub]); out[k]=float(roc_auc_score(te.y,gb.predict_proba(te[cols])[:,1])); print(k,round(out[k],4),flush=True)
# média de inadimplência por prazo no teste (padrão de contrato)
print(te.groupby(pd.cut(te.Term,[0,60,84,120,240,1000]),observed=True).y.agg(["size","mean"]).round(3))
json.dump(out,open("saida/diagnostico_gb.json","w"),ensure_ascii=False,indent=1)
