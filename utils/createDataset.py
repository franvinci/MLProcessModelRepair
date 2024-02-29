import numpy as np
import pandas as pd
import pm4py
from pm4py.algo.discovery.footprints import algorithm as footprints_discovery
from sklearn.feature_selection import VarianceThreshold


np.random.seed(72)

class CreateDataset:

    def __init__(self, real_train, real_val, sim):

        self.real_train =  real_train
        self.real_val = real_val
        self.sim = sim
        self.df_sim = pm4py.convert_to_dataframe(sim)
        self.df_sim_train = self.df_sim[self.df_sim['case:concept:name'].astype(int)<=len(self.real_train)]
        self.df_sim_val = self.df_sim[self.df_sim['case:concept:name'].astype(int)>len(self.real_train)]
        self.sim_train = pm4py.convert_to_event_log(self.df_sim_train)
        self.sim_val = pm4py.convert_to_event_log(self.df_sim_val)

        self.activities = list(pm4py.get_event_attribute_values(self.real_train, "concept:name").keys())
        self.activities_k = dict(zip(self.activities, range(len(self.activities))))

        self.features = self.getFeatureList()
        self.dfTrain = self.create_df(self.real_train, self.sim_train)
        self.dfVal = self.create_df(self.real_val, self.sim_val)


    def getCausalityRelationList(self):

        fp_real = footprints_discovery.apply(self.real_train, variant=footprints_discovery.Variants.ENTIRE_EVENT_LOG)
        fp_sim = footprints_discovery.apply(self.sim_train, variant=footprints_discovery.Variants.ENTIRE_EVENT_LOG)
        
        causalityRelations = list(set(list(fp_real['dfg'].keys()) + list(fp_sim['dfg'].keys())))

        return causalityRelations


    def getFeatureList(self):

        causalityRelations = self.getCausalityRelationList()
        features = [a1 + ' -> ' + a2 for (a1,a2) in causalityRelations]

        return features
    

    def create_df(self, real, sim):
        idx_features = dict(zip(self.features, range(len(self.features))))
        A = np.zeros((len(real) + len(sim), len(self.features)+1))

        for i, trace in enumerate(real):
            for j in range(len(trace)-1):
                f = trace[j]['concept:name'] + ' -> ' + trace[j+1]['concept:name']
                if f in self.features:
                    k = idx_features[f]
                    A[i, k] = 1
            A[i, -1] = 1

        for i, trace in enumerate(sim):
            for j in range(len(trace)-1):
                f = trace[j]['concept:name'] + ' -> ' + trace[j+1]['concept:name']
                if f in self.features:
                    k = idx_features[f]
                    A[i + len(sim), k] = 1
            A[i + len(sim), -1] = 0

        np.random.shuffle(A) 
        df_log = pd.DataFrame(A, columns=self.features+['is_real'])

        return df_log  


    def featureSelection_Variance(self, X_train, p=0.99):

        threshold = (p * (1 - p))
        self.selVar = VarianceThreshold(threshold=threshold)
        self.selVar.fit(X_train)
        selected_features = self.selVar.get_feature_names_out()

        return selected_features
    

    def featureSelection_Corr(self, X_train, thr=0.99):

        self.corr_train = X_train.corr()
        high_corr_train = (self.corr_train.abs() > thr)

        to_remove = []
        for f in self.corr_train.index:
            if f not in to_remove:
                r = [x for x in high_corr_train[f][high_corr_train[f]].index if x!=f]
                to_remove += r

        to_remove = list(set(to_remove))

        return to_remove


    def splitTrainVal(self, var_selection=True, corr_selection=True):

        X_train, y_train = self.dfTrain.iloc[:,:-1], self.dfTrain.iloc[:,-1]
        X_val, y_val = self.dfVal.iloc[:,:-1], self.dfVal.iloc[:,-1]

        if var_selection:
            selected_features = self.featureSelection_Variance(X_train)
            X_train = X_train[selected_features]
            X_val = X_val[selected_features]

        if corr_selection:
            to_remove = self.featureSelection_Corr(X_train)
            X_train = X_train.drop(to_remove, axis=1)
            X_val = X_val.drop(to_remove, axis=1)

        return X_train, X_val, y_train, y_val   