"""Recalcula accuracy/F1/precision/recall/fidelidade com scikit-learn a partir das previsões guardadas e compara com o registado."""
import json, glob, sys, os, numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
root=sys.argv[1]; mx=0; n=0
for u in sorted(glob.glob(root+'/raw/*/seed*/')):
    m=json.load(open(u+'mlp_original.json')); yt=np.array(m['predictions']['y_true']); om=np.array(m['predictions']['surrogate_predictions']); okey=list(m['predictions']['oracle_predictions'].keys())[0]
    for f in sorted(glob.glob(u+'*.json')):
        a=os.path.basename(f)[:-5]
        if a in('split','UNIT_SHA256'): continue
        d=json.load(open(f)); p=np.array(d['predictions']['surrogate_predictions']); r=d['row']
        assert (np.array(d['predictions']['y_true'])==yt).all(), 'instâncias de teste diferentes entre braços'
        chk={'accuracy_real_labels':accuracy_score(yt,p),'macro_f1_real_labels':f1_score(yt,p,average='macro'),'weighted_f1_real_labels':f1_score(yt,p,average='weighted'),
             'precision_macro_real_labels':precision_score(yt,p,average='macro',zero_division=0),'recall_macro_real_labels':recall_score(yt,p,average='macro')}
        if a!='mlp_original':
            assert (np.array(d['predictions']['oracle_predictions'][okey])==om).all(), 'oráculo diferente entre braços'
            chk['fidelity_to_oracle']=float(np.mean(p==om))
        for k,v in chk.items():
            if r.get(k) is not None: mx=max(mx,abs(v-r[k])); n+=1
print('valores verificados:',n,'| diferença absoluta máxima:',mx)
