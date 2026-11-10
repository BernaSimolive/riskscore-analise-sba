# -*- coding: utf-8 -*-
"""
Teste de censura/seleção do desfecho (nov/2026).
Pergunta: o forte efeito do prazo do contrato vem de a base SBA conter só empréstimos com desfecho já resolvido?
Ideia: repetir a análise só com empréstimos cujo vencimento contratual (início + prazo) ocorre até a data de corte dos dados.
Nesse subconjunto, todo empréstimo teve tempo de ser quitado ou baixado, e a inclusão depende só de variáveis conhecidas
na concessão (data e prazo), não do desfecho.
Como rodar:  python analise_censura.py  (com SBAnational.csv na mesma pasta). Saída: saida/censura.json
"""
import json, os
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
OUT="saida"; os.makedirs(OUT,exist_ok=True)
USE=["NAICS","ApprovalDate","ApprovalFY","DisbursementDate","Term","NoEmp","NewExist","FranchiseCode","UrbanRural","RevLineCr","LowDoc","MIS_Status","GrAppv"]
df=pd.read_csv("SBAnational.csv",dtype=str,encoding="latin-1",usecols=USE)
df["GrAppv"]=pd.to_numeric(df["GrAppv"].str.replace(r"[$,\s]","",regex=True),errors="coerce")
for c in ["NoEmp","Term","NewExist","UrbanRural","FranchiseCode"]: df[c]=pd.to_numeric(df[c],errors="coerce")
df["FY"]=pd.to_numeric(df["ApprovalFY"].str.replace("A","",regex=False),errors="coerce")
df=df[df["MIS_Status"].isin(["P I F","CHGOFF"])&df["NewExist"].isin([1,2])&(df.FY>=1990)&(df.FY<=2010)&(df.Term>0)&(df.GrAppv>0)&(df.NoEmp>=0)].copy()
def parse(s, fy):
    d=pd.to_datetime(s,format="%d-%b-%y",errors="coerce")
    return d.where(d.dt.year<=2030, d-pd.DateOffset(years=100))   # anos de 2 dígitos: 69-99 -> 19xx
df["inicio"]=parse(df["DisbursementDate"],df.FY).fillna(parse(df["ApprovalDate"],df.FY))
df["venc"]=df["inicio"]+pd.to_timedelta(df["Term"]*30.4375,unit="D")   # vencimento = início + prazo em meses
df["y"]=(df.MIS_Status=="CHGOFF").astype(int); df["novo"]=(df.NewExist==2).astype(int)
df["franq"]=(df.FranchiseCode>1).astype(int); df["lowdoc"]=(df.LowDoc=="Y").astype(int); df["rev"]=(df.RevLineCr=="Y").astype(int)
df["setor"]=pd.to_numeric(df.NAICS.str[:2],errors="coerce").fillna(0); df["urb"]=df.UrbanRural.fillna(0)
df["faixa"]=pd.cut(df.Term,[0,60,84,120,240,1000],labels=["até 60","61 a 84","85 a 120","121 a 240","mais de 240"])
sets={"só empresa":["NoEmp","novo","setor","urb","franq"],
      "empresa + prazo":["NoEmp","novo","setor","urb","franq","Term"],
      "empresa + valor":["NoEmp","novo","setor","urb","franq","GrAppv"],
      "empresa + contrato":["NoEmp","novo","setor","urb","franq","Term","GrAppv","lowdoc","rev"]}
def gb_auc(tr,te,cols):
    rng=np.random.default_rng(1); sub=rng.choice(len(tr),min(150000,len(tr)),replace=False)
    m=HistGradientBoostingClassifier(random_state=1,max_iter=80,categorical_features=[i for i,c in enumerate(cols) if c=="setor"] or None)
    m.fit(tr[cols].iloc[sub],tr.y.iloc[sub]); return float(roc_auc_score(te.y,m.predict_proba(te[cols])[:,1]))
def cenario(nome,d,fy_teste_max=2010):
    tr=d[d.FY<=2004]; te=d[(d.FY>=2005)&(d.FY<=fy_teste_max)]
    r={"n_treino":len(tr),"n_teste":len(te),"taxa_treino":float(tr.y.mean()),"taxa_teste":float(te.y.mean())}
    r["taxa_por_prazo_teste"]={str(k):{"n":int(v["size"]),"taxa":float(v["mean"])} for k,v in te.groupby("faixa",observed=True).y.agg(["size","mean"]).iterrows()}
    r["auc_prazo_isolado"]=float(roc_auc_score(te.y,-te.Term))            # risco = prazo menor
    ref=float(tr.NoEmp.quantile(0.95)); r["auc_score_porte"]=float(roc_auc_score(te.y,-np.minimum(te.NoEmp/ref,1.0)))
    X=lambda d_: np.c_[np.log1p(d_.NoEmp),d_.novo]; sc=StandardScaler().fit(X(tr)); lr=LogisticRegression(max_iter=500).fit(sc.transform(X(tr)),tr.y)
    r["auc_logit_mesmas_variaveis"]=float(roc_auc_score(te.y,lr.predict_proba(sc.transform(X(te)))[:,1]))
    r["gb"]={k:gb_auc(tr,te,c) for k,c in sets.items()}
    print(nome,json.dumps({k:v for k,v in r.items() if k!='taxa_por_prazo_teste'},ensure_ascii=False),flush=True)
    return r
res={}
res["original (sem filtro)"]=cenario("original",df)
for corte in ["2014-12-31","2013-12-31"]:
    m=df[df["venc"]<=pd.Timestamp(corte)]
    res[f"maturados até {corte}"]=cenario(corte,m); res[f"maturados até {corte}"]["fracao_mantida"]=float(len(m)/len(df))
# desenho 2: safras 2005-2007 e prazo até 84 meses (todos vencem até 2014 mesmo com início em 2007)
m2=df[(df.Term<=84)&(df.FY<=2007)]
res["safras 2005-2007, prazo até 84"]=cenario("safras 2005-2007, prazo<=84",m2,fy_teste_max=2007)
# frações mantidas por safra e faixa de prazo (corte 2014-12-31)
frac=df.assign(ok=(df["venc"]<=pd.Timestamp("2014-12-31")).astype(int)).groupby(["FY","faixa"],observed=True).ok.mean().unstack().loc[2003:]
res["fracao_maturada_por_safra_e_prazo"]=frac.round(3).reset_index().to_dict("records")
json.dump(res,open(f"{OUT}/censura.json","w",encoding="utf-8"),ensure_ascii=False,indent=1,default=float)
print("ok")
