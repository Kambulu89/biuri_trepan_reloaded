from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from core.ontology_semantic_graph import OntologySemanticGraph
from core.semantic_contribution_gate import SemanticContributionGate
from core.trepan_reloaded_historical import TrepanReloadedClassifier


class E:
    def __init__(self,name,**attrs):
        self.name=name
        for k,v in attrs.items(): setattr(self,k,v)

class O:
    def __init__(self):
        parent=E('Measurement')
        self.a=E('FeatureA',is_a=[parent],label=['feature_a'])
        self.b=E('FeatureB',is_a=[parent],label=['feature_b'])
        self.c=E('FeatureC',label=['feature_c'])
        self.parent=parent
    def classes(self): return [self.parent]
    def data_properties(self): return [self.a,self.b,self.c]
    def object_properties(self): return []
    def annotation_properties(self): return []
    def individuals(self): return []


def test_semantic_graph_is_dataset_agnostic_and_groups_related_features():
    onto=O()
    matches=[
        {'feature':'feature_a','entity_name':'FeatureA','accepted':True},
        {'feature':'feature_b','entity_name':'FeatureB','accepted':True},
        {'feature':'feature_c','entity_name':'FeatureC','accepted':True},
    ]
    g=OntologySemanticGraph.from_ontology(onto,accepted_matches=matches)
    assert g.primary_group('feature_a')=='Measurement'
    assert g.primary_group('feature_b')=='Measurement'
    assert g.relatedness('FeatureA','FeatureB')>0
    assert g.relatedness('FeatureA','FeatureC')==0
    m=np.asarray(g.feature_relatedness_matrix(['feature_a','feature_b','feature_c']))
    assert m.shape==(3,3) and m[0,1]>m[0,2]


def test_semantic_contribution_gate_separates_quality_from_observed_effect():
    quality={'accepted':True,'status':'VALID_DOMAIN_ONTOLOGY'}
    no_effect=SemanticContributionGate().evaluate(
        ontology_quality=quality,
        semantic_audit={'ontology_usage_rate':0.0,'semantic_decision_impact':0.0},
        comparison={'delta_oracle_fidelity':0.0,'delta_balanced_accuracy':0.0,'delta_macro_f1':0.0},
        original_metrics={'nodes':3},reloaded_metrics={'nodes':3},
    )
    assert not no_effect['accepted']
    assert no_effect['status']=='NO_OBSERVED_SEMANTIC_CONTRIBUTION'

    effect=SemanticContributionGate().evaluate(
        ontology_quality=quality,
        semantic_audit={'ontology_usage_rate':0.5,'semantic_decision_impact':0.2},
        comparison={'delta_oracle_fidelity':0.01,'delta_balanced_accuracy':0.0,'delta_macro_f1':0.0},
        original_metrics={'nodes':5},reloaded_metrics={'nodes':6},
    )
    assert effect['accepted']
    assert effect['status']=='SEMANTIC_CONTRIBUTION_SUPPORTED'


def test_relatedness_matrix_can_activate_reloaded_without_feature_weight_difference():
    class Oracle:
        def predict(self,X):
            X=np.asarray(X); return ((X[:,0]>0).astype(int) + (X[:,1]>0).astype(int) >= 1).astype(int)
    rng=np.random.RandomState(7); X=rng.normal(size=(120,3))
    rel=np.eye(3); rel[0,1]=rel[1,0]=1.0
    model=TrepanReloadedClassifier(max_nodes=5,max_depth=2,min_sample=120,max_queries=300,max_n=2,random_state=3)
    model.fit(X,oracle=Oracle(),feature_names=['a','b','c'],semantic_feature_weights=np.ones(3),semantic_relatedness_matrix=rel)
    assert model.reload_extension_active_
    assert model.semantic_audit_summary()['evaluated_splits']>=1


def test_production_bundle_roundtrip_without_owl(tmp_path):
    from core.production_training import train_production_dataframe
    from core.production_inference import load_production_bundle
    X,y=make_classification(n_samples=90,n_features=5,n_informative=4,n_redundant=0,random_state=13)
    df=pd.DataFrame(X,columns=[f'f{i}' for i in range(5)]); df['target']=np.where(y==1,'sim','nao')
    report=train_production_dataframe(df,target='target',out_dir=tmp_path,seed=9,owl_path=None)
    predictor=load_production_bundle(tmp_path)
    sample=df.drop(columns=['target']).iloc[:5]
    assert predictor.validate(sample)['valid']
    assert len(predictor.predict(sample,model='mlp'))==5
    assert len(predictor.predict(sample,model='trepan_original'))==5
    assert len(predictor.predict(sample,model='trepan_reloaded'))==5
    assert report['evaluation']['semantic_contribution_gate']['status']=='NO_ONTOLOGY'
