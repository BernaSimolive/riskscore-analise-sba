# Análise empírica de um score de crédito ponderado e simplificado (SBA National)

Código de análise do artigo "Avaliação empírica de um score de crédito ponderado e
simplificado para pequenas empresas" (Universidade São Judas Tadeu, Engenharia de Software).
Este repositório é separado do backend do sistema (RiskScoreTcc/pme-scoring).

## Base de dados
- Fonte original: Li, Mickel e Taylor (2018), "Should this loan be approved or denied?",
  Journal of Statistics Education, v. 26, n. 1, p. 55-66. DOI: 10.1080/10691898.2018.1434342.
- Cópia usada: release v1.0.0 de JuliusHmto/data-sources (arquivo SBAnational.zip), baixada em 19/10/2026.
- Arquivo SBAnational.csv (899.164 registros): SHA-256
  `0359128a0b7599e83e4c2e4dcdd781d9121a765237f98d9f0ec1ab3e7c522548`
- A base não está neste repositório. Baixe-a da fonte e confira o hash: `sha256sum SBAnational.csv`.

## Como rodar
Python 3.13, pandas 3.0.5, scikit-learn 1.9.1, scipy 1.18.1, matplotlib 3.11.2.
Coloque SBAnational.csv na mesma pasta dos scripts.

    python analise_sba.py          # modelos, AUC/Gini/KS, ICs (DeLong e bootstrap), equivalência
    python analise_censura.py      # teste de censura do prazo
    python diagnostico_prazo.py    # diagnóstico do boosting por grupo de variáveis

Os scripts gravam na pasta `saida/`; os resultados usados no artigo estão copiados em `resultados/` (resultados.json, censura.json, tabelas e figuras).
Semente fixa: 42 (modelos) e 2026 (bootstrap).

## Versão
Marque o commit usado no artigo com uma tag (por exemplo `v1.0-atcc`) e cite-a no artigo.
