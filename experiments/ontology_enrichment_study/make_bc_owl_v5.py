import sys
XS="http://www.w3.org/2001/XMLSchema#"
BASE="https://w3id.org/xai-ontologies/breast_cancer_wisconsin_diagnostic"
measures=[("radius","SizeGeometry"),("perimeter","SizeGeometry"),("area","SizeGeometry"),
          ("compactness","ContourIndentation"),("concavity","ContourIndentation"),("concave_points","ContourIndentation"),
          ("smoothness","BoundaryRegularity"),("symmetry","BoundaryRegularity"),("fractal_dimension","BoundaryRegularity"),
          ("texture","IntensityTexture")]
stats=[("Mean","mean","{m}_mean"),("StandardError","error","{m}_se"),("Worst","worst","{m}_worst")]
subfams=["SizeGeometry","ContourIndentation","BoundaryRegularity","IntensityTexture"]
def camel(m): return "".join(p.capitalize() for p in m.split("_"))
o=[f'''<?xml version="1.0" encoding="utf-8"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#" xmlns:owl="http://www.w3.org/2002/07/owl#" xmlns:xsd="{XS}" xmlns:dct="http://purl.org/dc/terms/"
         xml:base="{BASE}" xmlns="{BASE}#">

<owl:Ontology rdf:about="{BASE}">
  <dct:title rdf:datatype="{XS}string">Breast Cancer Wisconsin (Diagnostic) — ontologia de domínio enriquecida para o BIURI</dct:title>
  <dct:source rdf:resource="https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic"/>
  <rdfs:comment rdf:datatype="{XS}string">TBox de domínio com nomes UCI (campo_dataset do mapeo; zero instâncias; rótulos e resultados não consultados). Conhecimento de domínio codificado: (1) as 30 features são 10 medidas morfológicas do núcleo celular × 3 estatísticas — média (mean), erro padrão (error) e pior valor (worst, média dos três maiores valores); cada propriedade declara a sua measurementFamily e o seu statisticRole, o que permite ao BIURI relacionar a mesma característica entre estatísticas; (2) as medidas agrupam-se, segundo a descrição do dicionário do dataset, em geometria de tamanho (raio, perímetro, área), indentação do contorno (compacidade, concavidade, pontos côncavos), regularidade da fronteira (suavidade, simetria, dimensão fractal) e textura de intensidade; cada propriedade tem como domínio a classe «estatística × subfamília»; (3) todas as medidas são grandezas não negativas (faceta xsd:minInclusive 0). ESTATUTO: descrição citada de memória do dicionário público do dataset, NÃO verificada neste ambiente; revisão humana pendente.</rdfs:comment>
  <owl:versionInfo rdf:datatype="{XS}string">biuri-5.0</owl:versionInfo>
</owl:Ontology>

<owl:AnnotationProperty rdf:about="#measurementFamily"/>
<owl:AnnotationProperty rdf:about="#biuriColumnName"/>
<owl:AnnotationProperty rdf:about="#biuriIsTarget"/>
<owl:AnnotationProperty rdf:about="#biuriClassOrder"/>
<owl:AnnotationProperty rdf:about="#statisticRole"/>

<owl:Class rdf:about="#DiagnosisTarget"><rdfs:label rdf:datatype="{XS}string">diagnosis (target variable)</rdfs:label><biuriColumnName rdf:datatype="{XS}string">diagnosis</biuriColumnName><biuriIsTarget rdf:datatype="{XS}string">true</biuriIsTarget><biuriClassOrder rdf:datatype="{XS}string">M,B</biuriClassOrder></owl:Class>
<owl:Class rdf:about="#NuclearMeasurement"><rdfs:subClassOf rdf:resource="http://www.w3.org/2002/07/owl#Thing"/><rdfs:label rdf:datatype="{XS}string">nuclear measurement</rdfs:label></owl:Class>
''']
def disj(names, suffix):
    return {a:"".join(f'<owl:disjointWith rdf:resource="#{b}{suffix}"/>' for b in names[i+1:]) for i,a in enumerate(names)}
sn=[s for s,_,_ in stats]; d=disj(sn,"StatisticMeasurement")
for s in sn: o.append(f'<owl:Class rdf:about="#{s}StatisticMeasurement"><rdfs:subClassOf rdf:resource="#NuclearMeasurement"/>{d[s]}<rdfs:label rdf:datatype="{XS}string">{s} statistic measurement</rdfs:label></owl:Class>\n')
d=disj(subfams,"Measurement")
for f in subfams: o.append(f'<owl:Class rdf:about="#{f}Measurement"><rdfs:subClassOf rdf:resource="#NuclearMeasurement"/>{d[f]}<rdfs:label rdf:datatype="{XS}string">{f} measurement family</rdfs:label></owl:Class>\n')
for s,_,_ in stats:
    for f in subfams:
        o.append(f'<owl:Class rdf:about="#{s}{f}Measurement"><rdfs:subClassOf rdf:resource="#{s}StatisticMeasurement"/><rdfs:subClassOf rdf:resource="#{f}Measurement"/><rdfs:label rdf:datatype="{XS}string">{s} {f} measurement</rdfs:label></owl:Class>\n')
for s,role,pat in stats:
    for m,f in measures:
        col=pat.format(m=m)
        o.append(f'''
<owl:DatatypeProperty rdf:about="#has{s}{camel(m)}Value">
  <rdfs:domain rdf:resource="#{s}{f}Measurement"/>
  <rdfs:range><rdfs:Datatype><owl:onDatatype rdf:resource="{XS}decimal"/><owl:withRestrictions rdf:parseType="Collection"><rdf:Description><xsd:minInclusive rdf:datatype="{XS}decimal">0</xsd:minInclusive></rdf:Description></owl:withRestrictions></rdfs:Datatype></rdfs:range>
  <rdfs:label rdf:datatype="{XS}string">{col}</rdfs:label>
  <measurementFamily rdf:datatype="{XS}string">{m}</measurementFamily>
  <statisticRole rdf:datatype="{XS}string">{role}</statisticRole>
</owl:DatatypeProperty>
''')
o.append("\n</rdf:RDF>\n")
open(sys.argv[1],"w",encoding="utf-8").write("".join(o))
