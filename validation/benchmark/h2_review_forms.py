"""H2 (Doc. 54 §7): gera as TRÊS fichas de revisão separadas (domínio, engenharia de ontologias, metodologia) para a proposta ``german_credit/v0_draft``. Cada revisor recebe só o que é da sua
competência. Os campos de nome, data, referência de assinatura e decisão ficam VAZIOS: só pessoas designadas os podem preencher; esta ferramenta nunca declara revisores, assinaturas nem aprovações."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from validation.benchmark.h2_german_proposal import OUT, cards

BLANK = "_(a preencher por pessoa designada — vazio)_"


def header(role, title, scope):
    return (f"# Ficha de revisão — {title}\n\nProposta: `german_credit/v0_draft` (rascunho de IA, NÃO validado). Papel: **{role}**.\n"
            f"Âmbito desta ficha (só o que é da sua competência): {scope}\n\n"
            f"| Campo | Valor |\n|---|---|\n| Nome do revisor | {BLANK} |\n| Data | {BLANK} |\n| Referência de assinatura | {BLANK} |\n| Decisão global (`APPROVED` / `REJECTED` / `CHANGES_REQUESTED`) | {BLANK} |\n\n")


def domain_form(cs):
    t = header("DOMÍNIO", "especialista do domínio", "verdade e âmbito das afirmações, significado das categorias, classificação nominal/ordinal, fonte documental. NÃO avalia OWL nem metodologia.")
    t += "Fontes a confirmar (todas `NOT_VERIFIED`): dicionário de dados `german.doc` do UCI Statlog (atributos 1, 4, 6, 7, 17).\n\n| Ficha | Afirmação | Fonte (citação / localizador) | Verdadeira? | Nominal/ordinal confirmado? | Âmbito e exceções | Estado |\n|---|---|---|---|---|---|---|\n"
    for c in cs:
        t += f"| `{c.id}` | {c.statement_nl} | {c.source['citation']} / {c.source['locator']} | {BLANK} | {BLANK} | {BLANK} | `DOMAIN_REVIEW_PENDING` |\n"
    return t


def ontology_form(cs):
    t = header("ENGENHARIA DE ONTOLOGIAS", "engenheiro de ontologias", "forma OWL 2 DL, consistência, satisfatibilidade, inferências esperadas vs obtidas, matching, mundo aberto. NÃO avalia a verdade de domínio nem a metodologia.")
    t += "Resultados da máquina para conferência (HermiT: O2 consistente, 0 classes insatisfazíveis, 29 axiomas inferidos; matching 61/61): `docs/evidence/b1b4/b3q/h2_german_reasoner_check.json`.\n\n| Ficha | Axioma OWL | Inferências esperadas (escrever antes de consultar) | Conforme? | Observações (perfil DL, punning, mundo aberto) | Estado |\n|---|---|---|---|---|---|\n"
    for c in cs:
        t += f"| `{c.id}` | `{c.owl_axiom}` | {BLANK} | {BLANK} | {BLANK} | `ONTOLOGY_REVIEW_PENDING` |\n"
    return t


def methodology_form(cs):
    t = header("METODOLOGIA", "responsável pela metodologia experimental", "independência dos rótulos de teste, ausência de fuga, pré-registo, controlos, hashes, critérios de congelamento. NÃO avalia a verdade de domínio nem OWL.")
    t += "Verificações (preencher Sim/Não + observação):\n\n| Verificação | Resposta |\n|---|---|\n"
    for q in ["Os axiomas foram escritos sem consultar rótulos nem resultados de teste?", "A construção dos literais é independente de y (assinatura sem rótulos)?", "A ontologia de controlo (neutralizada, permutada, aleatória) tem tamanhos e estrutura comparáveis?",
              "O contrato de identidade (instâncias, oráculo, representação, recursos) permite cada comparação pretendida?", "Os hashes do OWL, das fichas e do código estão registados antes de qualquer avaliação?",
              "O conjunto é tratado como DEV exploratório (exposto), nunca confirmatório?", "Os limiares das portas foram congelados antes de avaliar variantes?"]:
        t += f"| {q} | {BLANK} |\n"
    t += f"\nFichas cobertas: {', '.join('`%s`' % c.id for c in cs)}. Estado: `METHODOLOGY_REVIEW_PENDING`.\n"
    return t


def main():
    d = OUT / "review_forms"
    d.mkdir(parents=True, exist_ok=True)
    cs = cards()
    (d / "domain_review.md").write_text(domain_form(cs), encoding="utf-8")
    (d / "ontology_review.md").write_text(ontology_form(cs), encoding="utf-8")
    (d / "methodology_review.md").write_text(methodology_form(cs), encoding="utf-8")
    print(sorted(p.name for p in d.iterdir()))


if __name__ == "__main__":
    main()
