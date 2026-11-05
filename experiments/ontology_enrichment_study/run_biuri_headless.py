"""Executa o fluxo INTERATIVO da GUI do BIURI em modo offscreen (sem alterar o código do BIURI): carrega ARFF (+OWL), treina, compara métricas e imprime o que a aplicação mostra."""
import os, sys, time, json, re
os.environ["QT_QPA_PLATFORM"]="offscreen"
sys.path.insert(0, "/home/user/biuri_trepan_reloaded")
from PyQt6.QtWidgets import QApplication, QMessageBox
msgs=[]
for n in ("warning","information","critical","question"):
    setattr(QMessageBox,n,staticmethod(lambda *a,_n=n,**k:(msgs.append((_n,str(a[1:3]))),QMessageBox.StandardButton.Ok)[1]))
app=QApplication([])
from gui.biuri_app_complete import BiuriApp
arff, owl, out = sys.argv[1], (sys.argv[2] if sys.argv[2]!="-" else None), sys.argv[3]
w=BiuriApp(); w.show()
texts=[]; _o=w.show_results
w.show_results=lambda t,*a,**k:(texts.append(str(t)),_o(t,*a,**k))[1]
w.load_data_from_file(arff)
if owl: print("ontologia carregada:", w.load_ontology_from_file(owl))
t0=time.time(); w.train_model()
while time.time()-t0<float(os.environ.get("BIURI_WAIT_S","3000")):
    app.processEvents(); time.sleep(0.05)
    wk=getattr(w,"_training_worker",None)
    if wk is not None and not wk.isRunning() and getattr(w,"mlp_model",None) is not None: break
for _ in range(60): app.processEvents(); time.sleep(0.05)
dur=time.time()-t0
try:
    w.compare_metrics(); 
    for _ in range(40): app.processEvents(); time.sleep(0.05)
except Exception as e: print("compare ERR",repr(e)[:200])
md=getattr(getattr(w,"metrics_widget",None),"metrics_visualizer",None)
models=getattr(md,"models_data",{}) if md else {}
full="\n".join(texts)
res={"arff":arff,"owl":owl,"train_seconds":round(dur,1),"mlp_trained":w.mlp_model is not None,"models":{k:{kk:vv for kk,vv in v.items() if isinstance(vv,(int,float,str,type(None)))} for k,v in models.items()},"messages":msgs[:8],"text":full}
json.dump(res,open(out,"w"),ensure_ascii=False,indent=1,default=str)
print("fim", round(dur,1), "modelos:", list(models))
w.close()
