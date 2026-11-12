import sys, os, glob, json, hashlib, subprocess, tempfile
ROOT = "/home/user/kambulu89/biuri_trepan_reloaded"
files = sorted(set(glob.glob(ROOT + "/data/**/*.owl", recursive=True) + glob.glob(ROOT + "/validation/**/*.owl", recursive=True)
                   + glob.glob(ROOT + "/experiments/**/*.owl", recursive=True)))
code = r'''
import sys, json, os
from owlready2 import *
p = sys.argv[1]
t = open(p, encoding="utf-8", errors="replace").read()
out = dict(xml_lang=t.count("xml:lang"), imports=t.count("owl:imports"), named_individual_tags=t.count("NamedIndividual"))
try:
    o = get_ontology(os.path.abspath(p)).load()
    out.update(loaded=True, classes=len(list(o.classes())), data_props=len(list(o.data_properties())), obj_props=len(list(o.object_properties())),
               annotation_props=len(list(o.annotation_properties())), individuals=len(list(o.individuals())),
               disjoint=len(list(o.disjoint_classes())),
               domains=sum(1 for q in o.data_properties() if q.domain), ranges=sum(1 for q in o.data_properties() if q.range),
               constrained_ranges=sum(1 for q in o.data_properties() for r in q.range if type(r).__name__ == "ConstrainedDatatype"),
               labels=sum(1 for q in o.data_properties() if q.label))
    try:
        with o: sync_reasoner_hermit(infer_property_values=False, debug=0)
        out.update(hermit="consistent", unsatisfiable=[c.name for c in default_world.inconsistent_classes()])
    except Exception as e:
        out.update(hermit="FAILED: " + type(e).__name__ + ": " + str(e)[:160])
except Exception as e:
    out.update(loaded=False, error=type(e).__name__ + ": " + str(e)[:200])
print("RESULT" + json.dumps(out))
'''
res = {}
for f in files:
    r = subprocess.run([sys.executable, "-c", code, f], capture_output=True, text=True, timeout=600)
    line = [l for l in r.stdout.splitlines() if l.startswith("RESULT")]
    d = json.loads(line[0][6:]) if line else {"loaded": False, "error": (r.stderr or "")[-300:]}
    d["sha256"] = hashlib.sha256(open(f, "rb").read()).hexdigest()[:16]
    res[os.path.relpath(f, ROOT)] = d
    print(os.path.relpath(f, ROOT), json.dumps(d), flush=True)
json.dump(res, open("/home/user/audit_logs/audit_ontologies.json", "w"), indent=1)
