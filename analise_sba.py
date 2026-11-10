# -*- coding: utf-8 -*-
"""
Análise exploratória e avaliação de um score do tipo RiskScore na base pública SBA National
(Li, Mickel & Taylor, 2018, Journal of Statistics Education, DOI 10.1080/10691898.2018.1434342).

Como rodar:  python analise_sba.py  (com SBAnational.csv na mesma pasta)
Saídas: resultados.json, tabelas em CSV e figuras PNG na pasta ./saida
IMPORTANTE: a base SBA NÃO possui faturamento nem histórico de inadimplência da empresa.
O "score proxy" aqui usa apenas porte (nº de funcionários) e tempo de atividade (negócio novo x existente).

Versão 2 (nov/2026): intervalos de confiança do AUC por DeLong e por bootstrap com 2.000 reamostragens, e teste da
diferença pareada entre o score proxy e a regressão logística com as mesmas variáveis (DeLong pareado, bootstrap
pareado e equivalência com margem declarada). As métricas pontuais e os modelos são os mesmos da versão 1.
"""
import json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.stats import rankdata, norm

SEED = 42
rng = np.random.default_rng(SEED)
rng_ci = np.random.default_rng(2026)      # gerador separado para os ICs (não altera o sorteio da amostra do boosting)
B_BOOT = 2000                              # reamostragens bootstrap
MARGENS = [0.01, 0.02]                     # margens de equivalência do AUC (declaradas a priori)
OUT = "saida"; os.makedirs(OUT, exist_ok=True)
plt.rcParams["font.family"] = "Liberation Serif"; plt.rcParams["font.size"] = 10

# ---------------- 1. Carga e limpeza ----------------
USE = ["NAICS","ApprovalFY","Term","NoEmp","NewExist","FranchiseCode","UrbanRural","RevLineCr","LowDoc","MIS_Status","GrAppv"]
df = pd.read_csv("SBAnational.csv", dtype=str, encoding="latin-1", usecols=USE)
N_RAW = len(df)
def money(s): return pd.to_numeric(s.str.replace(r"[$,\s]", "", regex=True), errors="coerce")
df["GrAppv"] = money(df["GrAppv"])
for c in ["NoEmp", "Term", "NewExist", "UrbanRural", "FranchiseCode"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["FY"] = pd.to_numeric(df["ApprovalFY"].str.replace("A", "", regex=False), errors="coerce")

df = df[df["MIS_Status"].isin(["P I F", "CHGOFF"])].copy()          # remove status ausente
N_STATUS = len(df)
df["default"] = (df["MIS_Status"] == "CHGOFF").astype(int)
df = df[df["NewExist"].isin([1, 2])].copy()                           # remove 0/ausente
df = df[(df["FY"] >= 1990) & (df["FY"] <= 2010)].copy()               # evita censura de empréstimos recentes
df = df[(df["NoEmp"] >= 0) & (df["Term"] > 0) & (df["GrAppv"] > 0)].copy()
df["novo"] = (df["NewExist"] == 2).astype(int)
df["franquia"] = (df["FranchiseCode"] > 1).astype(int)
df["lowdoc"] = (df["LowDoc"] == "Y").astype(int)
df["revline"] = (df["RevLineCr"] == "Y").astype(int)
df["setor"] = df["NAICS"].str[:2].fillna("00")
df["urb"] = df["UrbanRural"].fillna(0).astype(int).astype(str)
df["lnEmp"] = np.log1p(df["NoEmp"]); df["lnTerm"] = np.log(df["Term"]); df["lnGr"] = np.log(df["GrAppv"])
N_FINAL = len(df)

res = {"n_raw": N_RAW, "n_com_status": N_STATUS, "n_final": N_FINAL,
       "taxa_inadimplencia_final": float(df["default"].mean()),
       "n_inadimplentes_final": int(df["default"].sum()),
       "fy_min": int(df["FY"].min()), "fy_max": int(df["FY"].max())}

# ---------------- 2. Descritiva ----------------
def taxa(g):
    return pd.Series({"empresas": len(g), "inadimplentes": int(g["default"].sum()), "taxa": g["default"].mean()})
t_fy = df.groupby("FY").apply(taxa, include_groups=False).reset_index(); t_fy.to_csv(f"{OUT}/taxa_por_ano.csv", index=False)
t_novo = df.groupby("novo").apply(taxa, include_groups=False).reset_index()
t_novo["novo"] = t_novo["novo"].map({0: "Negócio existente", 1: "Negócio novo"})
t_novo.to_csv(f"{OUT}/taxa_por_tempo_atividade.csv", index=False)
bins = [-1, 4, 9, 19, 49, 1e9]; labs = ["0 a 4", "5 a 9", "10 a 19", "20 a 49", "50 ou mais"]
df["porte"] = pd.cut(df["NoEmp"], bins=bins, labels=labs)
t_porte = df.groupby("porte", observed=True).apply(taxa, include_groups=False).reset_index(); t_porte.to_csv(f"{OUT}/taxa_por_porte.csv", index=False)
res["desc_novo"] = t_novo.to_dict("records"); res["desc_porte"] = t_porte.to_dict("records")

# ---------------- 3. Separação temporal e aleatória ----------------
oot_train = df[df["FY"] <= 2004].copy(); oot_test = df[df["FY"] >= 2005].copy()
rnd_train, rnd_test = train_test_split(df, test_size=0.3, random_state=SEED, stratify=df["default"])
res["split_oot"] = {"treino_n": len(oot_train), "treino_taxa": float(oot_train["default"].mean()),
                    "teste_n": len(oot_test), "teste_taxa": float(oot_test["default"].mean())}
res["split_rnd"] = {"treino_n": len(rnd_train), "teste_n": len(rnd_test),
                    "teste_taxa": float(rnd_test["default"].mean())}

# ---------------- 4. Métricas ----------------
def metrics(y, risk):
    auc = roc_auc_score(y, risk); fpr, tpr, _ = roc_curve(y, risk)
    return {"auc": float(auc), "gini": float(2 * auc - 1), "ks": float(np.max(tpr - fpr))}
def boot_auc(y, risk, B=30):   # legado (v1): mantido só para preservar o sorteio aleatório da amostra do boosting; os ICs do artigo vêm de inferencia()
    y = np.asarray(y); risk = np.asarray(risk); n = len(y); out = []
    for _ in range(B):
        i = rng.integers(0, n, n); out.append(roc_auc_score(y[i], risk[i]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]

# ---------------- 4b. Inferência: DeLong, bootstrap e diferença pareada ----------------
def delong(y, P):
    """DeLong rápido (Sun e Xu, 2014). P: dict nome -> risco (maior = mais inadimplente). Retorna AUCs e covariância."""
    y = np.asarray(y).astype(bool); names = list(P); k = len(names)
    X = np.vstack([np.asarray(P[n], dtype=float) for n in names])
    pos, neg = X[:, y], X[:, ~y]; m, n = pos.shape[1], neg.shape[1]
    tx = np.vstack([rankdata(r) for r in pos]); ty = np.vstack([rankdata(r) for r in neg]); tz = np.vstack([rankdata(r) for r in np.hstack([pos, neg])])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n; v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.atleast_2d(np.cov(v01)) / m + np.atleast_2d(np.cov(v10)) / n
    return names, aucs, cov
def cell_auc(g, n0, n1):
    """AUC a partir de contagens por grupo de escore (g = índice de grupo ordenado por risco crescente)."""
    g0 = np.bincount(g, weights=n0); less = np.cumsum(g0) - g0
    return float((n1 * (less[g] + 0.5 * g0[g])).sum() / (n1.sum() * n0.sum()))
def inferencia(te, P, ref_pair):
    """ICs por DeLong para todos os modelos e bootstrap (B_BOOT) + diferença pareada para o par (score, logística equivalente)."""
    y = te["default"].values; names, aucs, cov = delong(y, P); z = norm.ppf(0.975); out = {}
    for i, nme in enumerate(names):
        se = float(np.sqrt(cov[i, i])); out[nme] = {"auc": float(aucs[i]), "se_delong": se, "ic95_delong": [float(aucs[i] - z * se), float(aucs[i] + z * se)]}
    a, b = ref_pair; ia, ib = names.index(a), names.index(b)
    d = float(aucs[ia] - aucs[ib]); sed = float(np.sqrt(cov[ia, ia] + cov[ib, ib] - 2 * cov[ia, ib]))
    dif = {"par": [a, b], "diferenca_auc": d, "se_delong": sed, "ic95_delong": [d - z * sed, d + z * sed], "ic90_delong": [d - norm.ppf(0.95) * sed, d + norm.ppf(0.95) * sed],
           "p_delong": float(2 * (1 - norm.cdf(abs(d) / sed)))}
    # bootstrap pareado: os dois escores são funções de (NoEmp, novo); reamostrar observações = multinomial sobre as células
    cel = te[["NoEmp", "novo"]].copy(); cel["y"] = y; cel["ra"] = np.asarray(P[a]); cel["rb"] = np.asarray(P[b])
    assert cel.groupby(["NoEmp", "novo"])[["ra", "rb"]].nunique().max().max() == 1
    t = cel.groupby(["NoEmp", "novo", "y"]).agg(n=("y", "size"), ra=("ra", "first"), rb=("rb", "first")).reset_index()
    keys = t[["NoEmp", "novo"]].drop_duplicates().reset_index(drop=True); K = len(keys)
    idx = t.merge(keys.reset_index(), on=["NoEmp", "novo"])["index"].values
    n0c = np.zeros(K); n1c = np.zeros(K); n0c[idx[t["y"].values == 0]] = t.loc[t["y"] == 0, "n"].values; n1c[idx[t["y"].values == 1]] = t.loc[t["y"] == 1, "n"].values
    rr = t.drop_duplicates(["NoEmp", "novo"]).merge(keys.reset_index(), on=["NoEmp", "novo"]).sort_values("index")
    ga = np.unique(rr["ra"].values, return_inverse=True)[1]; gb_ = np.unique(rr["rb"].values, return_inverse=True)[1]
    assert abs(cell_auc(ga, n0c, n1c) - aucs[ia]) < 1e-9 and abs(cell_auc(gb_, n0c, n1c) - aucs[ib]) < 1e-9
    N = int(n0c.sum() + n1c.sum()); prob = np.concatenate([n0c, n1c]) / N
    cnt = rng_ci.multinomial(N, prob, size=B_BOOT); A = np.empty(B_BOOT); Bv = np.empty(B_BOOT)
    for j in range(B_BOOT):
        A[j] = cell_auc(ga, cnt[j, :K].astype(float), cnt[j, K:].astype(float)); Bv[j] = cell_auc(gb_, cnt[j, :K].astype(float), cnt[j, K:].astype(float))
    D = A - Bv; pc = lambda v, q: float(np.percentile(v, q))
    out[a]["ic95_bootstrap"] = [pc(A, 2.5), pc(A, 97.5)]; out[b]["ic95_bootstrap"] = [pc(Bv, 2.5), pc(Bv, 97.5)]
    dif["ic95_bootstrap"] = [pc(D, 2.5), pc(D, 97.5)]; dif["ic90_bootstrap"] = [pc(D, 5), pc(D, 95)]; dif["B"] = B_BOOT
    dif["equivalencia"] = {str(mg): {"ic90_delong_dentro": bool(-mg < dif["ic90_delong"][0] and dif["ic90_delong"][1] < mg),
                                      "ic90_bootstrap_dentro": bool(-mg < dif["ic90_bootstrap"][0] and dif["ic90_bootstrap"][1] < mg)} for mg in MARGENS}
    return out, dif
P = {}

# ---------------- 5. Score proxy tipo RiskScore ----------------
# Score = 1000 * (wT*T + wR*R + wD*D), pesos somando 1. Aqui: T = tempo (existente=1, novo=0);
# R = porte (nº funcionários / referência máxima, teto 1) como substituto de faturamento;
# D (histórico de inadimplência) não existe na base => componente constante (não altera a ordenação).
def score_proxy(d, wT, ref):
    R = np.minimum(d["NoEmp"] / ref, 1.0); T = 1 - d["novo"]
    return 1000 * (wT * T + (1 - wT) * R)
def run_scheme(train, test, nome):
    ref = float(train["NoEmp"].quantile(0.95))
    grid = np.round(np.arange(0, 1.0001, 0.1), 1)
    sens = []
    for w in grid:
        sens.append({"wT": float(w), "auc_treino": roc_auc_score(train["default"], -score_proxy(train, w, ref)),
                     "auc_teste": roc_auc_score(test["default"], -score_proxy(test, w, ref))})
    sens = pd.DataFrame(sens); sens.to_csv(f"{OUT}/sensibilidade_pesos_{nome}.csv", index=False)
    return ref, sens
out = {}
for nome, (tr, te) in {"oot": (oot_train, oot_test), "rnd": (rnd_train, rnd_test)}.items():
    ref, sens = run_scheme(tr, te, nome)
    W_REF = 0.5                                                           # configuração de referência (hipótese)
    W_FIT = float(sens.loc[sens["auc_treino"].idxmax(), "wT"])            # peso escolhido só com o treino
    r = {"ref_emp": ref, "w_ref": W_REF, "w_ajustado_treino": W_FIT}
    for tag, w in [("proxy_pesos_ref", W_REF), ("proxy_pesos_ajustados", W_FIT)]:
        s = score_proxy(te, w, ref); r[tag] = metrics(te["default"], -s); P[tag] = -s.values
        r[tag]["auc_ic95_legado_B30"] = boot_auc(te["default"], -s)
    # faixas de risco (terços do score no treino)
    s_tr = score_proxy(tr, W_FIT, ref); q1, q2 = s_tr.quantile([1/3, 2/3])
    s_te = score_proxy(te, W_FIT, ref)
    faixa = np.where(s_te <= q1, "Alto", np.where(s_te <= q2, "Médio", "Baixo"))
    tb = pd.DataFrame({"faixa": faixa, "default": te["default"].values}).groupby("faixa").agg(
        empresas=("default", "size"), inadimplentes=("default", "sum"), taxa=("default", "mean")).reindex(["Baixo", "Médio", "Alto"]).reset_index()
    tb.to_csv(f"{OUT}/faixas_{nome}.csv", index=False); r["faixas"] = tb.to_dict("records"); r["cortes"] = [float(q1), float(q2)]
    # modelos de referência
    X1 = ["lnEmp", "novo"]
    X2 = ["lnEmp", "novo", "lnTerm", "lnGr", "franquia", "lowdoc", "revline"]
    def design(d, cols, cats, cols_ref=None):
        M = d[cols].copy().astype(float)
        for c in cats: M = pd.concat([M, pd.get_dummies(d[c], prefix=c, dtype=float)], axis=1)
        return M
    cats = ["urb", "setor"]
    Mtr1, Mte1 = design(tr, X1, []), design(te, X1, [])
    Mtr2 = design(tr, X2, cats); Mte2 = design(te, X2, cats).reindex(columns=Mtr2.columns, fill_value=0.0)
    for tag, (A, B) in {"logit_equivalente": (Mtr1, Mte1), "logit_ampliado": (Mtr2, Mte2)}.items():
        sc = StandardScaler().fit(A); lr = LogisticRegression(max_iter=500).fit(sc.transform(A), tr["default"])
        p = lr.predict_proba(sc.transform(B))[:, 1]; r[tag] = metrics(te["default"], p); r[tag]["auc_ic95_legado_B30"] = boot_auc(te["default"], p)
        P[tag] = p
        if tag == "logit_ampliado": p_logit = p
        if tag == "logit_equivalente": p_eq = p
    sub = rng.choice(len(Mtr2), size=min(150000, len(Mtr2)), replace=False)
    gb = HistGradientBoostingClassifier(random_state=SEED, max_iter=80).fit(Mtr2.iloc[sub], tr["default"].iloc[sub])
    pg = gb.predict_proba(Mte2)[:, 1]; r["gboost_ampliado"] = metrics(te["default"], pg); r["gboost_ampliado"]["auc_ic95_legado_B30"] = boot_auc(te["default"], pg)
    P["gboost_ampliado"] = pg
    r["inferencia"], r["diferenca_score_x_logit"] = inferencia(te, P, ("proxy_pesos_ajustados", "logit_equivalente")); P = {}
    print("inferencia", nome, r["diferenca_score_x_logit"], flush=True)
    out[nome] = r
    print("concluido", nome, flush=True)
    json.dump(out, open(f"{OUT}/parcial.json","w",encoding="utf-8"), ensure_ascii=False, default=float)
    if nome == "oot":
        roc_data = {"RiskScore proxy (pesos ajustados)": -s_te.values, "Regressão logística (mesmas variáveis)": p_eq,
                    "Regressão logística ampliada": p_logit, "Gradient boosting ampliado": pg}
        y_oot = te["default"].values; sens_oot = sens
res["resultados"] = out

# ---------------- 6. Figuras ----------------
fig, ax = plt.subplots(figsize=(6.3, 3.2), dpi=200)
ax.bar(t_fy["FY"], t_fy["taxa"] * 100, color="0.45", edgecolor="black", linewidth=0.4)
ax.axvline(2004.5, color="black", ls="--", lw=0.8); ax.text(2004.7, ax.get_ylim()[1] * 0.93, "teste fora do tempo →", fontsize=8)
ax.set_xlabel("Ano fiscal de aprovação"); ax.set_ylabel("Inadimplência observada (%)"); ax.set_xticks(range(1990, 2011, 2))
for s in ["top", "right"]: ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_taxa_por_ano.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.6, 4.2), dpi=200)
styles = [("-", 1.6, "black"), ("--", 1.2, "0.3"), (":", 1.6, "0.2"), ("-.", 1.2, "0.5")]
for (nm, risk), (ls, lw, col) in zip(roc_data.items(), styles):
    fpr, tpr, _ = roc_curve(y_oot, risk); ax.plot(fpr, tpr, ls=ls, lw=lw, color=col, label=nm)
ax.plot([0, 1], [0, 1], color="0.7", lw=0.8); ax.set_xlabel("Taxa de falsos positivos"); ax.set_ylabel("Taxa de verdadeiros positivos")
ax.legend(fontsize=7, loc="lower right", frameon=False)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_roc_teste_fora_do_tempo.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.6, 3.0), dpi=200)
ax.plot(sens_oot["wT"], sens_oot["auc_treino"], "-o", color="black", ms=3, lw=1, label="Treino (1990–2004)")
ax.plot(sens_oot["wT"], sens_oot["auc_teste"], "--s", color="0.4", ms=3, lw=1, label="Teste (2005–2010)")
ax.set_xlabel("Peso do tempo de atividade (restante no porte)"); ax.set_ylabel("AUC-ROC"); ax.legend(fontsize=8, frameon=False)
for s in ["top", "right"]: ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_sensibilidade_pesos.png"); plt.close(fig)

json.dump(res, open(f"{OUT}/resultados.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
print("ok")
