import matplotlib.pyplot as plt
import numpy as np
def plot_feature_importance(feature_importances_ranked):
    feature_names_25 = [i[0] for i in feature_importances_ranked[:10]]
    y_ticks = np.arange(0, len(feature_names_25))
    x_axis = [i[1] for i in feature_importances_ranked[:10]]
    plt.figure(figsize = (30, 14))
    plt.barh(feature_names_25, x_axis)
    plt.title('Random Forest Feature Importance (Top 10)',fontdict= {'fontname':'Comic Sans MS','fontsize' : 20})
    plt.xlabel('Features',fontdict= {'fontsize' : 16})
    plt.show()