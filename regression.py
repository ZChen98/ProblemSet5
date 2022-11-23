"""
Justin Chen
Problem Set 5
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.discrete_model import Logit

def first_exercise():
    """
    Imports and cleans wage data;
    Generates visualization, regression table
    """
    # 1.1
    w_df = pd.read_csv('wage.csv')
    print(w_df.isnull().sum())

    # 1.2
    sns.pairplot(w_df[['wage', 'educ', 'numdep', 'tenure', 'exper']])

    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    ax1.scatter(w_df['educ'], w_df['wage'])
    ax1.set_title('educ and wage')
    ax1.set_xlabel('educ')
    ax1.set_ylabel('wage')

    ax2.scatter(w_df['educ'], w_df['lwage'])
    ax2.set_title('educ and lwage')
    ax2.set_xlabel('educ')
    ax2.set_ylabel('lwage')

    ax3.scatter(w_df['exper'], w_df['lwage'])
    ax3.set_title('exper and lwage')
    ax3.set_xlabel('exper')
    ax3.set_ylabel('lwage')

    ax4.scatter(w_df['tenure'], w_df['lwage'])
    ax4.set_title('tenure and lwage')
    ax4.set_xlabel('tenure')
    ax4.set_ylabel('lwage')

    plt.show()
    plt.tight_layout()

    # 1.5
    mod = smf.ols(formula='wage ~ educ + exper + tenure + married + female + profocc + west',
                    data=w_df)
    res = mod.fit()
    print(res.summary())

    # 1.8
    hypothetical = pd.DataFrame({"educ": [281], "exper": [270], "tenure": [255],
    "married": [1], "female": [0], "profocc": [1], "west": [1]})
    hypothetical_prediction = res.predict(hypothetical)
    print(hypothetical_prediction)

def second_exercise():
    """
    Imports and cleans diabetes data;
    Generates visualization, regression table
    """
    # 2.1
    d_df = pd.read_csv('diabetes.csv')
    print(d_df.isnull().sum())

    diabetes_diag = {'neg': 0, 'pos':1}
    d_df['diabetes_int'] = d_df['diabetes'].map(diabetes_diag)

    # 2.2
    sns.lmplot(x='pedigree', y='diabetes_int', data=d_df).set(
        title='LPM Plot: pedigree and diabetes')
    sns.lmplot(x='pedigree', y='diabetes_int', data=d_df, logistic= True).set(
        title='Logistic Plot: pedigree and diabetes')

    sns.lmplot(x='pregnant', y='diabetes_int', data=d_df).set(
        title='LPM Plot: pregnant and diabetes')
    sns.lmplot(x='pregnant', y='diabetes_int', data=d_df, logistic= True).set(
        title='Logistic Plot: pregnant and diabetes')

    sns.lmplot(x='glucose', y='diabetes_int', data=d_df).set(
        title='LPM Plot: glucose and diabetes')
    sns.lmplot(x='glucose', y='diabetes_int', data=d_df, logistic= True).set(
        title='Logistic Plot: glucose and diabetes')

    sns.lmplot(x='mass', y='diabetes_int', data=d_df).set(
        title='LPM Plot: mass and diabetes')
    sns.lmplot(x='mass', y='diabetes_int', data=d_df, logistic= True).set(
        title='Logistic Plot: mass and diabetes')

    sns.lmplot(x='age', y='diabetes_int', data=d_df).set(
        title='LPM Plot: age and diabetes')
    sns.lmplot(x='age', y='diabetes_int', data=d_df, logistic= True).set(
        title='Logistics Plot: age and diabetes')
    plt.show()

    # 2.5
    dfy = d_df['diabetes_int']
    dfx = sm.add_constant(d_df[['pregnant', 'glucose', 'mass', 'pedigree', 'age']])
    mod = Logit(dfy, dfx)
    res = mod.fit()
    print(res.summary())

    # 2.7
    d_df[['pregnant', 'glucose', 'mass', 'pedigree', 'age']].describe()

    hypothetical = [[1, 2, 119, 33.2, 0.4495, 27],
    [1, 5, 143, 37.1, 0.687, 36], [1, 1, 99, 28.4, 0.26975, 23]]
    hypothetical_prediction = res.predict(hypothetical)
    print(hypothetical_prediction)
