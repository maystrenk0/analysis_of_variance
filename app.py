import scipy.stats as stats
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from patsy.contrasts import Sum

df = pd.read_csv('Sample-Data-Guinea-Pig-Tooth-Growth.csv', sep=" ", usecols=['len', 'dose'])
ax = sns.boxplot(x='dose', y='len', data=df, whis=np.inf)
ax = sns.stripplot(x='dose', y='len', data=df, color=".3")
plt.show()

model = ols('dose ~ len', data=df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)
print(anova_table)
