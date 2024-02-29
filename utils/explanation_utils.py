import pandas as pd
import matplotlib.pyplot as plt

def plot_shap_by_value(shap_values, counts, shap_thr=0.05, show=True, save_to=None):
    s_values = shap_values.values[:,:,1]
    base_value = shap_values.base_values[0,1]
    data = shap_values.data
    feature_names = shap_values.feature_names

    shap_feature_value_dict = dict()
    for feature in feature_names:
        idx = feature_names.index(feature)
        s_values_f = s_values[:,idx]
        data_f = data[:,idx]
        if len(s_values_f[data_f==0]) > 0:
            shap_feature_value_dict[feature+' = 0'] = (s_values_f[data_f==0] * counts[data_f==0]).sum() / counts[data_f==0].sum()
        if len(s_values_f[data_f>0]) > 0:
            shap_feature_value_dict[feature+' > 0'] = (s_values_f[data_f>0] * counts[data_f>0]).sum() / counts[data_f>0].sum()
   
    shap_feature_value = pd.Series(shap_feature_value_dict)
    shap_feature_value = shap_feature_value[shap_feature_value.abs()>=shap_thr]
    shap_feature_value = shap_feature_value.sort_values().round(2)

    c = ['r' if x <0 else 'g' for x in shap_feature_value.values]

    if c == []:
        return shap_feature_value

    if show:
        plt.figure(figsize=(12,5))
        ax = (shap_feature_value).plot.barh(color=c)
        plt.title(f'Shap Values. Base value: {base_value.round(2)}')
        plt.bar_label(ax.containers[0], label_type='edge', fontsize=10)
        plt.show()

    if save_to:
        plt.figure(figsize=(12,5))
        ax = (shap_feature_value).plot.barh(color=c)
        plt.title(f'Shap Values. Base value: {base_value.round(2)}')
        plt.bar_label(ax.containers[0], label_type='edge', fontsize=10)
        plt.savefig(save_to, bbox_inches='tight')

    return shap_feature_value