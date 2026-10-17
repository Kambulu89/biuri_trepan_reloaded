"""H2 (Docs. 51–52): contrato axioma → mecanismo m-of-n e validação de fichas de axioma. Só ``validation/``. NENHUM mecanismo está integrado no núcleo.

* ``CONTRACTS``: para cada tipo de axioma, os 10 pontos do contrato (significado de domínio; representação OWL; o que o reasoner infere; caminho até aos atributos; facto semântico tipado;
  efeito nos candidatos; natureza (restrição lógica vs preferência vs enriquecimento de representação); medida de influência; controlo; riscos). ``validate_contracts`` falha se faltar algum ponto.
* ``AxiomCard`` / ``validate_card``: ficha de axioma. Regras duras: nada é «validado» por uma IA; aprovações exigem revisor humano nomeado, data e referência de assinatura; ``HARD_CONSTRAINT`` só
  para axiomas universais com as três aprovações; proibido derivar axiomas de correlações dos dados; declaração de independência face à avaliação obrigatória."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

CONTRACT_FIELDS = ("domain_meaning", "owl_representation", "reasoner_inference", "arff_path", "fact_kind", "candidate_effect", "nature", "influence_measure", "neutralizing_control", "risks")
NATURES = ("LOGICAL_CONSTRAINT", "HEURISTIC_PREFERENCE", "REPRESENTATION_ENRICHMENT", "NOT_APPLICABLE")
STATUSES = ("IMPLEMENTED_PROTOTYPE_VALIDATION", "DESIGN_ONLY", "REJECTED_AS_UNIVERSAL")


@dataclass(frozen=True)
class AxiomContract:
    axiom_type: str
    domain_meaning: str
    owl_representation: str
    reasoner_inference: str
    arff_path: str
    fact_kind: str
    candidate_effect: str
    nature: str
    influence_measure: str
    neutralizing_control: str
    risks: str
    open_world_rule: str
    status: str


CONTRACTS: Dict[str, AxiomContract] = {c.axiom_type: c for c in [
    AxiomContract("T1_DEFINED_CLASS_UNION", "uma categoria definida como união de níveis (p. ex. «poupança ≥ 500») ou um agrupamento taxonómico (p. ex. «veículo» = carro novo ∪ usado)",
                  "classe definida `C ≡ L1 ⊔ … ⊔ Lk`, níveis `Li ⊑ Variável`", "subsunção `Li ⊑ C` e, entre classes definidas, `C_k ⊑ C_j` quando o conjunto de níveis de C_k ⊆ C_j (inferência de HermiT, não declarada)",
                  "atributo ARFF (coluna 0/1 de um nível) → entidade (propriedade de dados ou classe de nível) pelo matching P3 → conceitos que a subsumem (`ATTRIBUTE_APPLIES_TO_CONCEPT`, derivação ASSERTED ou INFERRED_BY_REASONER)",
                  "ATTRIBUTE_APPLIES_TO_CONCEPT (+ CONCEPT_SUBSUMPTION); extensão do conceito = conjunto de atributos", "gera um LITERAL derivado binário `OR(colunas da extensão)` que passa a competir como candidato simples e como literal de m-of-n",
                  "REPRESENTATION_ENRICHMENT", "(a) o literal derivado entra no conjunto de candidatos gerados; (b) é escolhido num split; (c) contrafactual: ablação do axioma (a classe passa a primitiva) remove exatamente esse literal e regista a mudança de decisão",
                  "O3 neutralizada (classe primitiva, mesmo nome, sem definição); H permutada (ordem trocada); I união aleatória de tamanho igual; J limiares aprendidos dos dados",
                  "literal redundante com colunas existentes quando a extensão tem 1 elemento; tautologia quando cobre todos os níveis; custo de candidatos; enviesamento se a taxonomia refletir o rótulo", "mundo aberto: ausência de nível ≠ falso; o literal só se cria para colunas binárias mapeadas", "IMPLEMENTED_PROTOTYPE_VALIDATION"),
    AxiomContract("T2_ORDINAL_ORDER", "ordem entre níveis de uma variável ordinal documentada (poupança <100 < 100–500 < 500–1000 < ≥1000)", "cadeia de classes cumulativas `AtLeast_k ≡ L_k ⊔ … ⊔ L_m` (OWL 2 DL; sem propriedades transitivas sobre indivíduos)",
                  "`AtLeast_{k+1} ⊑ AtLeast_k` e `Li ⊑ AtLeast_k` para i ≥ k, todos INFERIDOS", "idêntico a T1 (as classes cumulativas são classes definidas)", "ATTRIBUTE_APPLIES_TO_CONCEPT + CONCEPT_SUBSUMPTION inferidas",
                  "literais de limiar cumulativo `nível ≥ k` (aninhados); permite regras m-of-n monótonas sobre variáveis ordinais one-hot", "REPRESENTATION_ENRICHMENT",
                  "como T1 e ainda: a ordem do axioma deve ser a ordem do literal escolhido", "H = ordem permutada (OWL válido, semântica errada); O3 = classes cumulativas primitivas; J = ordenação dos níveis pela taxa de classe no TREINO (engenharia convencional)",
                  "níveis não ordinais (desconhecido, «sem conta») têm de ficar de fora (erro lógico se incluídos); ordem pode ser convenção da codificação, não do domínio; o rótulo pode correlacionar com a ordem (fuga se a ordem se escolher pelos dados)", "ordem declarada só vale para níveis listados", "IMPLEMENTED_PROTOTYPE_VALIDATION"),
    AxiomContract("T3_DISJOINTNESS", "níveis de uma variável categórica são mutuamente exclusivos", "`AllDisjointClasses(L1…Lk)`", "consistência/insatisfazibilidade; instâncias em duas classes tornam a ontologia inconsistente",
                  "mapeamento coluna→classe de nível; só válido se a coluna for o indicador de pertença", "CONCEPT_DISJOINTNESS (permitted_use NONE hoje)", "poda de m-of-n com m ≥ 2 sobre literais mutuamente disjuntos (insatisfazível)",
                  "LOGICAL_CONSTRAINT", "n.º de candidatos podados; contrafactual: sem o axioma, o candidato existe", "O3 sem disjunção",
                  "redundante com a codificação one-hot (RECOVERABLE_FROM_DATA/encoding): não é conhecimento externo; nunca usar para validar ou descartar linhas", "NÃO converter em validação fechada de linhas; só poda candidatos logicamente insatisfazíveis dada a semântica da coluna", "DESIGN_ONLY"),
    AxiomContract("T4_VALUE_BOUND", "limite de domínio de um atributo (p. ex. idade ≥ 18 na população do estudo; contagens 0–16)", "restrição de tipo de dados com facetas (`xsd:minInclusive`/`maxInclusive`) numa `DatatypeProperty`", "nenhuma inferência nova",
                  "propriedade de dados mapeada ao atributo", "VALUE_BOUND", "restringe limiares candidatos e a plausibilidade das consultas ao oráculo (amostragem) ao domínio válido", "LOGICAL_CONSTRAINT (só em geração de consultas/limiares, nunca em linhas de treino)",
                  "fração de consultas ou limiares fora do domínio antes/depois; contrafactual: sem o limite", "O3 sem facetas; H limites trocados entre atributos (falsos: só como controlo negativo explícito, não como restrição dura)",
                  "limite como convenção do dataset, não do domínio; risco de recorte de valores legítimos; mundo aberto", "limite vale no âmbito declarado; não se aplica a linhas observadas", "DESIGN_ONLY"),
    AxiomContract("T5_ORDER_BETWEEN_ATTRIBUTES", "relação de ordem definicional entre medidas (p. ex. «pior» ≥ «média» por definição)", "axioma de domínio sobre propriedades de dados; em OWL 2 DL só como anotação formal ou regra externa (SWRL não-DL) — ver limitação", "nenhuma em OWL 2 DL puro",
                  "pares de atributos mapeados", "ORDER_RELATION (REGULARITY_ONLY até revisão)", "filtra consultas ao oráculo que violam a ordem e pode gerar literais de diferença", "LOGICAL_CONSTRAINT (consultas) / HEURISTIC_PREFERENCE",
                  "violações de consultas antes/depois; contrafactual: sem o filtro", "O3 sem a relação; H ordem entre pares errados só como controlo negativo declarado", "regularidade empírica ≠ axioma universal (Doc. 39: razões não valem para agregados); OWL 2 DL não exprime ordem entre valores de propriedades de dados de forma nativa",
                  "só com fonte e revisão; nunca derivada da correlação", "DESIGN_ONLY"),
    AxiomContract("T6_SUBPROPERTY_EQUIVALENCE", "atributos que medem a mesma grandeza (unidades diferentes) ou especialização de uma medida", "`SubDataPropertyOf`, `EquivalentDataProperties`", "propriedades equivalentes colapsam; subpropriedade herda domínio/âmbito",
                  "atributos mapeados a propriedades equivalentes", "PROPERTY_EQUIVALENCE / PROPERTY_SUBSUMPTION", "de-duplicação de literais redundantes (poupa candidatos) e agrupamento de literais intermutáveis", "HEURISTIC_PREFERENCE", "candidatos removidos por redundância; contrafactual",
                  "O3 sem equivalência; H equivalências trocadas", "colapso errado se a equivalência for aproximada; perda de informação", "equivalência só se declarada", "DESIGN_ONLY"),
    AxiomContract("T7_OBJECT_PROPERTY_ROLE", "papéis entre conceitos (mede, parte-de, causa)", "`ObjectProperty` (+ domínio/âmbito, caraterísticas)", "propagação por propriedades transitivas/cadeias", "indireto: atributo→conceito→relação",
                  "(sem facto tipado ainda)", "ponderação de relatedness e expansão de unidades de pesquisa", "HEURISTIC_PREFERENCE", "alteração da ordenação das unidades; contrafactual", "O3 sem a propriedade; I relações aleatórias de grau igual",
                  "relatedness constante hoje (hub `DatatypeProperty`); risco de confundir relação OWL com correlação", "mundo aberto", "DESIGN_ONLY"),
    AxiomContract("T8_CARDINALITY", "contagens admissíveis (exatamente um nível por variável)", "`ExactCardinality`/`FunctionalDataProperty`", "inconsistência em violação", "indireto", "(não suportado)", "limite para `n` de m-of-n sobre literais da mesma variável",
                  "LOGICAL_CONSTRAINT", "candidatos com n excessivo podados", "O3 sem cardinalidade", "redundante com one-hot; semântica OWL de cardinalidade ≠ validação de linhas", "NÃO converter em validação de linhas", "DESIGN_ONLY"),
    AxiomContract("T9_SPATIAL_ADJACENCY", "vizinhança na grelha de pixels (tipo de facto novo)", "relação de objetos simétrica entre atributos-como-entidades", "fecho de vizinhança", "atributo→entidade-pixel→vizinhos", "SPATIAL_ADJACENCY (inexistente)",
                  "unidades de pesquisa locais (m-of-n sobre vizinhança)", "HEURISTIC_PREFERENCE", "unidades geradas/escolhidas; contrafactual", "H adjacência permutada; I grafo aleatório de grau igual", "adjacência recuperável por correlação com n suficiente", "mundo aberto", "DESIGN_ONLY"),
]}


def validate_contracts() -> List[str]:
    problems = []
    for t, c in CONTRACTS.items():
        for f in CONTRACT_FIELDS + ("open_world_rule", "status"):
            if not str(getattr(c, f, "")).strip():
                problems.append(f"{t}: campo em falta {f}")
        if not c.nature.split(" ")[0] in NATURES:
            problems.append(f"{t}: natureza inválida")
        if c.status not in STATUSES:
            problems.append(f"{t}: estado inválido")
    return problems


REVIEW_ROLES = ("domain", "ontology", "methodology")


@dataclass
class AxiomCard:
    id: str
    statement_nl: str
    owl_axiom: str
    entities: List[str]
    axiom_type: str
    source: Optional[Dict[str, str]] = None              # {"citation": ..., "locator": ..., "verified": bool}
    scope_and_validity: str = ""
    universality: str = "REGULARITY_ONLY"                # UNIVERSAL | CONDITIONAL | REGULARITY_ONLY
    epistemic_status: str = "HYPOTHESIS"                 # LOGICALLY_ENTAILED | DOMAIN_DOCUMENTED | HYPOTHESIS
    permitted_use: str = "PREFERENCE"                    # NONE | PREFERENCE | HARD_CONSTRAINT
    authored_by: str = "AI_DRAFT"                        # AI_DRAFT | HUMAN_EXPERT
    derived_from_data: bool = False
    independence_from_eval: bool = True
    reviews: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {r: {"status": f"{r.upper()}_REVIEW_PENDING"} for r in REVIEW_ROLES})


def validate_card(c: AxiomCard) -> List[str]:
    p: List[str] = []
    if c.axiom_type not in CONTRACTS:
        p.append("tipo de axioma sem contrato")
    if not (c.id and c.statement_nl and c.owl_axiom and c.entities):
        p.append("campos obrigatórios em falta")
    if c.derived_from_data:
        p.append("PROIBIDO: axioma derivado de correlações/observações dos dados")
    if not c.independence_from_eval:
        p.append("independência face à avaliação não declarada")
    if c.epistemic_status != "HYPOTHESIS":
        if not c.source or not c.source.get("citation") or not c.source.get("locator"):
            p.append("estado documentado sem fonte com localizador")
    if c.authored_by == "AI_DRAFT":
        if c.epistemic_status == "DOMAIN_DOCUMENTED" and not (c.source or {}).get("verified"):
            p.append("rascunho de IA não pode ser DOMAIN_DOCUMENTED sem verificação humana da fonte")
        if c.permitted_use == "HARD_CONSTRAINT":
            p.append("rascunho de IA não pode ser restrição dura")
    for role in REVIEW_ROLES:
        r = c.reviews.get(role, {})
        st = r.get("status", "")
        if st.endswith("APPROVED") or st == "APPROVED":
            if c.authored_by == "AI_DRAFT" and not (r.get("reviewer") and r.get("date") and r.get("signature_ref")):
                p.append(f"aprovação '{role}' sem revisor, data e referência de assinatura")
            elif not (r.get("reviewer") and r.get("date") and r.get("signature_ref")):
                p.append(f"aprovação '{role}' sem revisor, data e referência de assinatura")
    if c.permitted_use == "HARD_CONSTRAINT":
        if c.universality != "UNIVERSAL":
            p.append("restrição dura exige axioma UNIVERSAL")
        if not all(c.reviews.get(r, {}).get("status") == "APPROVED" and c.reviews[r].get("reviewer") for r in REVIEW_ROLES):
            p.append("restrição dura exige as três revisões humanas aprovadas")
    return p
