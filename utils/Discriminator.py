from catboost import CatBoostClassifier
from utils.metrics import Accuracy
import shap
from utils.explanation_utils import plot_shap_by_value

class Discriminator:
    def __init__(self, data_creator, catboost_iterations=150, catboost_depth=3, catboost_lr=1e-2):
        self.data_creator = data_creator
        self.X_train, self.X_val, self.y_train, self.y_val = self.data_creator.splitTrainVal()
        self.model = CatBoostClassifier(iterations = catboost_iterations, depth = catboost_depth, random_seed=72, learning_rate=catboost_lr, task_type="GPU", devices='0:1', verbose=False)

    def fit(self):
        self.model.fit(self.X_train, self.y_train, eval_set = (self.X_val, self.y_val))
        return self
    
    # def eval_similarity(self):
    #     y_train_pred = self.model.predict(self.X_train)
    #     y_val_pred = self.model.predict(self.X_val)
    #     return SimilarityMetric(self.y_train, y_train_pred), SimilarityMetric(self.y_val, y_val_pred)
    
    def eval_accuracy(self):
        y_train_pred = self.model.predict(self.X_train)
        y_val_pred = self.model.predict(self.X_val)
        return Accuracy(self.y_train, y_train_pred), Accuracy(self.y_val, y_val_pred)
    
    def explain(self, p_conf, show=True, save_to=None):
        y_predict_prob = self.model.predict_proba(self.X_train)
        y_predict_prob = y_predict_prob[:,1]
        p_conf = p_conf
        y_predict = ((y_predict_prob > 0.5 - p_conf) & (y_predict_prob < 0.5 + p_conf))*0.5
        y_predict = y_predict + (y_predict_prob >= 0.5 + p_conf)

        X_train_True = self.X_train[(y_predict == self.y_train).values]
        X_train_True_unique = X_train_True.groupby(list(X_train_True.columns)).size().reset_index(name='Count')
        explainer = shap.Explainer(self.model.predict_proba, X_train_True_unique.iloc[:,:-1], seed=72)
        shap_values = explainer(X_train_True_unique.iloc[:,:-1])

        shap_feature_value = plot_shap_by_value(shap_values, X_train_True_unique['Count'].values, p_conf, show, save_to=save_to)
        base_value = shap_values.base_values[0,1]
        return shap_feature_value, base_value
    
    def recommendations(self, p_conf=0, corr_thr=0.99, show=True, save_to=None):
        shap_feature_value, _ = self.explain(p_conf, show, save_to)
        corr_train = self.data_creator.corr_train
        recommendations_dict = dict()
        recommendations_skip = dict()
        for x in shap_feature_value.abs().sort_values(ascending=False).index:
            if x in recommendations_dict.keys():
                continue
            p = shap_feature_value[x]
            if ' = 0' in x:
                f = x.split(' = ')[0]
                corr_features_p = corr_train[f][corr_train[f]>corr_thr].index
                corr_features_n = corr_train[f][corr_train[f]<-corr_thr].index

                # corr_features_p_1 = [feat.split(' -> ')[1] for feat in corr_features_p]
                # if (p > p_conf) and ('END' in corr_features_p_1):
                #     continue

                for f_corr in corr_features_p:
                    if p<-p_conf:
                        recommendations_dict[f_corr] = (p, 'skip')
                    # elif p>p_conf:
                    #     recommendations_skip[f_corr] = (p, 'skip_b')
                for f_corr in corr_features_n:
                    if p>p_conf:
                        recommendations_dict[f_corr] = (p, 'skip')
                    # elif p<-p_conf:
                    #     recommendations_skip[f_corr] = (p, 'skip_b')
            if ' > 0' in x:
                f = x.split(' > ')[0]
                corr_features_p = corr_train[f][corr_train[f]>corr_thr].index
                corr_features_n = corr_train[f][corr_train[f]<-corr_thr].index

                # corr_features_p_1 = [feat.split(' -> ')[1] for feat in corr_features_p]
                # if (p < -p_conf) and ('END' in corr_features_p_1):
                #     continue

                for f_corr in corr_features_p:
                    if p>p_conf:
                        recommendations_dict[f_corr] = (p, 'skip')
                    # elif p<-p_conf:
                    #     recommendations_skip[f_corr] = (p, 'skip_b')
                for f_corr in corr_features_n:
                    if p<-p_conf:
                        recommendations_dict[f_corr] = (p, 'skip')
                    # elif p>p_conf:
                    #     recommendations_skip[f_corr] = (p, 'skip_b')

        recommendations = recommendations_dict | recommendations_skip

        recommendations = dict(sorted(recommendations.items(), key=lambda x: abs(x[1][0]), reverse=True))

        return recommendations