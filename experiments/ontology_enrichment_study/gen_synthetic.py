"""Gera datasets sintéticos com estrutura semântica latente + ontologias (correta, permutada, aleatória).

Honestidade metodológica
- SINTÉTICO: prova de capacidade do mecanismo, NÃO evidência sobre datasets reais.
- A ontologia é escrita a partir do DESENHO do gerador (estrutura declarada), nunca a partir dos rótulos amostrados.
- Controlos: 'perm' (mesma estrutura de tamanhos, colunas trocadas entre grupos), 'rand' (partição aleatória de
  tamanhos arbitrários), 'none' (sem ontologia).
Uso: python gen_synthetic.py <outdir> <seed>
"""
import sys, random, os
import numpy as np

XS = "http://www.w3.org/2001/XMLSchema#"
BASE = "https://example.org/biuri/synthetic"

def _hdr(title, comment):
    return f'''<?xml version="1.0" encoding="utf-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" xmlns:owl="http://www.w3.org/2002/07/owl#" xmlns:xsd="{XS}"
         xml:base="{BASE}" xmlns="{BASE}#">
<owl:Ontology rdf:about="{BASE}">
  <rdfs:label rdf:datatype="{XS}string">{title}</rdfs:label>
  <rdfs:comment rdf:datatype="{XS}string">{comment}</rdfs:comment>
</owl:Ontology>
'''

def _camel(s): return "".join(p.capitalize() for p in s.split("_"))

# ---------------------------------------------------------------- S1: grupos numéricos latentes
STEMS = ["thermal", "vibration", "pressure", "flow", "acoustic", "voltage", "current", "humidity", "torque", "speed",
         "level", "conductivity", "oxygen", "light", "strain", "position", "displacement", "density", "viscosity", "turbidity"]
K = 6  # sensores por subsistema

def make_s1(seed, n=250):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, len(STEMS)))
    cols, names = [], []
    for g, stem in enumerate(STEMS):
        for k in range(1, K + 1):
            cols.append(1.0 * z[:, g] + rng.normal(size=n))
            names.append(f"{stem}_sensor_{k}")
    X = np.column_stack(cols)
    # rótulo depende APENAS de factores latentes de 2 subsistemas (estrutura do desenho)
    y = (z[:, 0] + z[:, 3] + 0.3 * rng.normal(size=n) > 0).astype(int)
    return X, y, names, {stem: [f"{stem}_sensor_{k}" for k in range(1, K + 1)] for stem in STEMS}

def arff_numeric(path, X, y, names, relation):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"@relation {relation}\n\n")
        for nme in names: f.write(f"@attribute {nme} numeric\n")
        f.write("@attribute failure_state {normal,fault}\n\n@data\n")
        for row, lab in zip(X, y):
            f.write(",".join(f"{v:.6f}" for v in row) + "," + ("fault" if lab else "normal") + "\n")

def owl_groups(path, groups, status, title):
    o = [_hdr(title, f"TBox sintético sem instâncias. Estatuto do conhecimento: {status}. "
                     "Agrupamentos escritos a partir do desenho do gerador; rótulos amostrados não consultados.")]
    o.append('<owl:Class rdf:about="#SensorMeasurement"><rdfs:label rdf:datatype="%sstring">sensor measurement</rdfs:label></owl:Class>\n' % XS)
    gn = list(groups)
    for i, g in enumerate(gn):
        dis = "".join(f'<owl:disjointWith rdf:resource="#{_camel(h)}SubsystemMeasurement"/>' for h in gn[i + 1:])
        o.append(f'<owl:Class rdf:about="#{_camel(g)}SubsystemMeasurement"><rdfs:subClassOf rdf:resource="#SensorMeasurement"/>{dis}'
                 f'<rdfs:label rdf:datatype="{XS}string">{g} subsystem measurement</rdfs:label></owl:Class>\n')
    for g, cs in groups.items():
        for c in cs:
            o.append(f'<owl:DatatypeProperty rdf:about="#has{_camel(c)}Value"><rdfs:domain rdf:resource="#{_camel(g)}SubsystemMeasurement"/>'
                     f'<rdfs:range rdf:resource="{XS}double"/><rdfs:label rdf:datatype="{XS}string">{c}</rdfs:label></owl:DatatypeProperty>\n')
    o.append("</rdf:RDF>\n")
    open(path, "w", encoding="utf-8").write("".join(o))

def permuted(groups, rng):
    cols = [c for cs in groups.values() for c in cs]; rng.shuffle(cols)
    out, i = {}, 0
    for g, cs in groups.items():
        out[g] = cols[i:i + len(cs)]; i += len(cs)
    return out

def random_partition(groups, rng):
    cols = [c for cs in groups.values() for c in cs]; rng.shuffle(cols)
    out, i, gi = {}, 0, 0
    while i < len(cols):
        size = rng.randint(2, 10); out[f"randgroup{gi}"] = cols[i:i + size]; i += size; gi += 1
    return {g: cs for g, cs in out.items() if len(cs) >= 1}

# ---------------------------------------------------------------- S2: hierarquia categórica
FAMILIES = ["valve", "pump", "bearing", "sensorhead", "coupling", "filter"]
PER = 4

def make_s2(seed, n=300):
    rng = np.random.default_rng(seed)
    codes = [f"{fam}_{chr(65 + j)}{j + 1}" for fam in FAMILIES for j in range(PER)]
    fam_of = {c: c.split("_")[0] for c in codes}
    risk = {fam: v for fam, v in zip(FAMILIES, rng.normal(size=len(FAMILIES)) * 1.5)}
    idx = rng.integers(0, len(codes), size=n)
    code = [codes[i] for i in idx]
    load = rng.normal(size=n); age = rng.normal(size=n); temp = rng.normal(size=n); run = rng.normal(size=n)
    score = np.array([risk[fam_of[c]] for c in code]) + 0.8 * load + 0.3 * rng.normal(size=n)
    y = (score > 0).astype(int)
    return code, load, age, temp, run, y, codes, fam_of

def arff_s2(path, code, load, age, temp, run, y, codes):
    with open(path, "w", encoding="utf-8") as f:
        f.write("@relation component_failure\n\n@attribute component_code {" + ",".join(codes) + "}\n")
        for nm in ("load_index", "age_index", "temperature_index", "runtime_index"): f.write(f"@attribute {nm} numeric\n")
        f.write("@attribute failure_state {normal,fault}\n\n@data\n")
        for i in range(len(y)):
            f.write(f"{code[i]},{load[i]:.6f},{age[i]:.6f},{temp[i]:.6f},{run[i]:.6f},{'fault' if y[i] else 'normal'}\n")

def owl_s2(path, fam_of, codes, status, title):
    fams = list(dict.fromkeys(fam_of[c] for c in codes))
    o = [_hdr(title, f"Taxonomia de componentes (vocabulário de valores, sem linhas do dataset). Estatuto: {status}.")]
    o.append(f'<owl:Class rdf:about="#Component"><rdfs:label rdf:datatype="{XS}string">component</rdfs:label></owl:Class>\n')
    for fam in fams:
        o.append(f'<owl:Class rdf:about="#{_camel(fam)}Family"><rdfs:subClassOf rdf:resource="#Component"/><rdfs:label rdf:datatype="{XS}string">{fam} family</rdfs:label></owl:Class>\n')
    for c in codes:
        o.append(f'<owl:NamedIndividual rdf:about="#{c}"><rdf:type rdf:resource="#{_camel(fam_of[c])}Family"/><rdfs:label rdf:datatype="{XS}string">{c}</rdfs:label></owl:NamedIndividual>\n')
    o.append(f'<owl:DatatypeProperty rdf:about="#hasComponentCode"><rdfs:domain rdf:resource="#Component"/><rdfs:range rdf:resource="{XS}string"/><rdfs:label rdf:datatype="{XS}string">component_code</rdfs:label></owl:DatatypeProperty>\n')
    for nm in ("load_index", "age_index", "temperature_index", "runtime_index"):
        o.append(f'<owl:DatatypeProperty rdf:about="#has{_camel(nm)}"><rdfs:domain rdf:resource="#Component"/><rdfs:range rdf:resource="{XS}double"/><rdfs:label rdf:datatype="{XS}string">{nm}</rdfs:label></owl:DatatypeProperty>\n')
    o.append("</rdf:RDF>\n")
    open(path, "w", encoding="utf-8").write("".join(o))

def main(out, seed):
    os.makedirs(out, exist_ok=True)
    rng = random.Random(seed * 7919)
    X, y, names, groups = make_s1(seed)
    arff_numeric(f"{out}/s1_seed{seed}.arff", X, y, names, "synthetic_subsystem_sensors")
    owl_groups(f"{out}/s1_seed{seed}_correct.owl", groups, "DECLARED_DESIGN (synthetic)", "S1 correct subsystem grouping")
    owl_groups(f"{out}/s1_seed{seed}_perm.owl", permuted(groups, rng), "NEGATIVE_CONTROL permuted", "S1 permuted grouping")
    owl_groups(f"{out}/s1_seed{seed}_rand.owl", random_partition(groups, rng), "NEGATIVE_CONTROL random partition", "S1 random grouping")
    code, load, age, temp, run, y2, codes, fam_of = make_s2(seed)
    arff_s2(f"{out}/s2_seed{seed}.arff", code, load, age, temp, run, y2, codes)
    owl_s2(f"{out}/s2_seed{seed}_correct.owl", fam_of, codes, "DECLARED_DESIGN (synthetic)", "S2 correct taxonomy")
    cs = list(codes); shuf = cs[:]; rng.shuffle(shuf)
    owl_s2(f"{out}/s2_seed{seed}_perm.owl", {c: fam_of[s] for c, s in zip(cs, shuf)}, codes, "NEGATIVE_CONTROL permuted taxonomy", "S2 permuted taxonomy")
    rf = {c: f"rand{rng.randint(0, 5)}" for c in codes}
    owl_s2(f"{out}/s2_seed{seed}_rand.owl", rf, codes, "NEGATIVE_CONTROL random taxonomy", "S2 random taxonomy")
    print("ok", out, seed)

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
