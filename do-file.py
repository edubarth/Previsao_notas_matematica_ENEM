# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:31:49 2020

@author: Eduardo
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pylab as pyl
import seaborn as sns
from pylab import scatter, show, title, xlabel, ylabel, plot, contour

## importando os dados
test = pd.read_csv("E:/Meu drive/desafio codenation/testfiles/test.csv")
train = pd.read_csv("E:/Meu drive/desafio codenation/testfiles/train.csv")

## limpeza da base train
train.columns
test.columns
train.head(10)

train=train[['NU_INSCRICAO', 'CO_UF_RESIDENCIA', 'SG_UF_RESIDENCIA', 'NU_IDADE',
       'TP_SEXO', 'TP_COR_RACA', 'TP_NACIONALIDADE', 'TP_ST_CONCLUSAO',
       'TP_ANO_CONCLUIU', 'TP_ESCOLA', 'TP_ENSINO', 'IN_TREINEIRO',
       'TP_DEPENDENCIA_ADM_ESC', 'IN_BAIXA_VISAO', 'IN_CEGUEIRA', 'IN_SURDEZ',
       'IN_DISLEXIA', 'IN_DISCALCULIA','IN_DEFICIENCIA_AUDITIVA', 'IN_SURDO_CEGUEIRA', 
       'IN_DEFICIENCIA_FISICA', 'IN_DEFICIENCIA_MENTAL', 'IN_DEFICIT_ATENCAO','IN_AUTISMO',
       'IN_VISAO_MONOCULAR', 'IN_OUTRA_DEF', 'IN_IDOSO', 'TP_PRESENCA_CN', 
       'TP_PRESENCA_CH', 'TP_PRESENCA_LC',
       'CO_PROVA_CN', 'CO_PROVA_CH', 'CO_PROVA_LC', 'CO_PROVA_MT',
       'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC','NU_NOTA_MT', 'TP_LINGUA',
       'TP_STATUS_REDACAO', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3',
       'NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_REDACAO', 'Q001', 'Q002',
       'Q006', 'Q024', 'Q025', 'Q026', 'Q027', 'Q047']]

## dropar as variaveis que são inteiras 0
train.sum()
test.sum()
train.drop(columns=['IN_AUTISMO', 'IN_CEGUEIRA', 'IN_SURDO_CEGUEIRA'], inplace=True)

## verificar e dropar varlores nulos
    train.isnull().sum()
    test.isnull().sum()
    
    train.dropna(subset = ["NU_NOTA_MT"], inplace=True)
    train.dropna(subset = ["NU_NOTA_CN"], inplace=True)
    
    test.dropna(subset = ["NU_NOTA_CN"], inplace=True)
    test.dropna(subset = ["NU_NOTA_LC"], inplace=True)



## remover notas 0
        mask=(train['NU_NOTA_MT']==0)
        mask1=(train['NU_NOTA_LC']==0)
        mask2=(train['NU_NOTA_CN']==0)
        mask3=(train['NU_NOTA_CH']==0)

        train=train.loc[~mask]
        train=train.loc[~mask1]
        train=train.loc[~mask2]
        train=train.loc[~mask3]


## criar variavel deficiencia
    defi=['IN_BAIXA_VISAO', 'IN_SURDEZ','IN_DISLEXIA', 'IN_DISCALCULIA','IN_DEFICIENCIA_AUDITIVA','IN_DEFICIENCIA_FISICA', 'IN_DEFICIENCIA_MENTAL', 'IN_DEFICIT_ATENCAO','IN_VISAO_MONOCULAR', 'IN_OUTRA_DEF']
       
    for i in defi:
         train[i]=train[i].replace('',0)
        
    for j in defi:
        train['deficiencia']=train[j]+train['deficiencia']
        
    train['deficiencia'].unique()
    ## substituir os repetidos
    train['deficiencia']=train['deficiencia'].replace([2,3],1)

## converter dummys para inteiros

dummy5=['Q024','Q047']
dummy8=['Q001', 'Q002']

    train['Q025']=train['Q025'].replace(['A','B'],[0,1])
    train['Q026']=train['Q026'].replace(['A','B','C'],[0,1,2])
    for i in dummy5:
        train[i]=train[i].replace(['A','B','C','D','E'],[0,1,2,3,4]) 
        
    for i in dummy8:
        train[i]=train[i].replace(['A','B','C','D','E','F','G','H'],[0,1,2,3,4,5,6,7])  
    
    train['Q027']=train['Q027'].replace(['A','B','C','D','E','F','G','H','I','J','K','L','M'],[0,1,2,3,4,5,6,7,8,9,10,11,12])
    train['Q006']=train['Q006'].replace(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q'],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])

    test['Q025']=test['Q025'].replace(['A','B'],[0,1])
    test['Q026']=test['Q026'].replace(['A','B','C'],[0,1,2])
    for i in dummy5:
        test[i]=test[i].replace(['A','B','C','D','E'],[0,1,2,3,4]) 
        
    for i in dummy8:
        test[i]=test[i].replace(['A','B','C','D','E','F','G','H'],[0,1,2,3,4,5,6,7])  
    
    test['Q027']=test['Q027'].replace(['A','B','C','D','E','F','G','H','I','J','K','L','M'],[0,1,2,3,4,5,6,7,8,9,10,11,12])
    test['Q006']=test['Q006'].replace(['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q'],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])


## correlação
   correlacao = train.corr()
    var_teste = ['Q001', 'Q002', 'Q006', 'Q024', 'Q025', 'Q026', 'Q027', 'Q047','deficiencia','TP_DEPENDENCIA_ADM_ESC','TP_COR_RACA','NU_IDADE','IN_TREINEIRO','NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_REDACAO','NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3','NU_NOTA_COMP4', 'NU_NOTA_COMP5', 'NU_NOTA_MT']
    matriz_corre_teste=train[var_teste].corr()
    ## POR MEIO DA MATRIZ DE CORRELAÇÃO PODEMOS IDENTIFICAR AS VARIAVEIS CORRELACIONADAS COM A NOSSA VARIAVEL DE INTERESSE (NU_NOTA_MT)
    ## PORTANTO, IREI CONSIDERAR APENAS AS QUE OBTIVERAM UM VALOR DE CORRELAÇÃO >= 0,15 EM MÓDULO: ('Q001', 'Q002', 'Q006', 'Q024', 'Q025','Q047', 'TP_COR_RACA', 'NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_REDACAO')
    ## APESAR DA ALTA CORRELAÇÃO COM NU_NOTA_MT, IREI DESCONSIDERAR AS VARIAVEIS: 'TP_DEPENDENCIA_ADM_ESC','NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3','NU_NOTA_COMP4', 'NU_NOTA_COMP5'. POIS ESTAS APRESENTAM ALTA CORRELAÇÃO COM OUTRAS VARIAVEIS DESEJADAS, O QUE PODERIA GERAR UM PROBLEMA DE AUTOCORRELAÇÃO NO MODELO.
    var=['Q001', 'Q002', 'Q006', 'Q024', 'Q025','Q047','TP_COR_RACA','NU_NOTA_CN','NU_NOTA_CH','NU_NOTA_LC','NU_NOTA_REDACAO']
    matriz_corre=train[var].corr()
    ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(matriz_corre,  annot=True, annot_kws={"size": 10})

## tirar valor 0 ou nulos
    
    train_final=train.drop(columns=['TP_DEPENDENCIA_ADM_ESC','NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3',
                                    'NU_NOTA_COMP4', 'NU_NOTA_COMP5','Q026', 'Q027','NU_IDADE','IN_TREINEIRO',
                                    'TP_ENSINO'])
    

nulos=train_final.isnull()
train_final.isnull().sum()

    test = test.loc[
      (test['NU_NOTA_CN'].notnull())  & (test['NU_NOTA_CN'] != 0) & (test['NU_NOTA_CH'].notnull())      & (test['NU_NOTA_CH'] != 0) 
    & (test['NU_NOTA_LC'].notnull())  & (test['NU_NOTA_LC'] != 0) & (test['NU_NOTA_REDACAO'].notnull()) & (test['NU_NOTA_REDACAO'] != 0)    
]

    test_final=test.drop(columns=['TP_DEPENDENCIA_ADM_ESC','NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3',
                                    'NU_NOTA_COMP4', 'NU_NOTA_COMP5','Q026', 'Q027','NU_IDADE','IN_TREINEIRO',
                                    'TP_ENSINO'])


test_final.isnull().sum()


## estatistica descritiva
train.sum()
test.sum()
train.mean()
train.max()

## boxplot
        plt.boxplot(train['NU_NOTA_MT'])

# histograma
        plt.hist(train['NU_NOTA_MT'])
        x0=train_final['NU_NOTA_CN'].fillna(0)
        x1=test['NU_NOTA_CN'].fillna(0)
        sns.distplot(x0)
        sns.distplot(x1)
        plt.legend(labels=['TRAIN','TEST'], ncol=2, loc='upper left')
        

## plots
    ## matematica X idade
        plt.scatter(train['Q006'], train['NU_NOTA_MT'])
        plt.title('Gráfico de correlação Nota Matemática X Idade')
        plt.xlabel('Idade')
        plt.ylabel('Nota Matemática')
        plt.show()
    
## modelo
        y_train=train_final['NU_NOTA_MT']
        x_train=train_final[var]
        x_test=test_final[var]
    
    ## normalizar os dados
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()  
        x_train = sc.fit_transform(x_train)  
        x_test = sc.transform(x_test)

    ## modelo regressão linear multipla
from sklearn.linear_model import LinearRegression

regressor_linear = LinearRegression().fit(x_train, y_train)
y_pred_linear_test=regressor_linear.predict(x_test)
y_pred_linear_train = regressor_linear.predict(x_train) 
print('MAE:', mean_absolute_error(y_train, y_pred_linear_train))

    
    ## modelo de Random Forest    
        from sklearn.ensemble import RandomForestRegressor
        regressor = RandomForestRegressor( 
            criterion='mae', 
            max_depth=8,
            max_leaf_nodes=None,
            min_impurity_split=None,
            min_samples_leaf=1,
            min_samples_split=2,
            min_weight_fraction_leaf=0.0,
            n_estimators= 500,
            n_jobs=-1,
            random_state=0,
            verbose=0,
            warm_start=False)
        

regressor.fit(x_train, y_train)

y_pred_test = regressor.predict(x_test)
y_pred_train = regressor.predict(x_train)

from sklearn.metrics import mean_absolute_error
print('MAE:', mean_absolute_error(y_train, y_pred_train))


plt.plot(y_train, color = 'red', label = 'Real data')
plt.plot(y_pred_train, color = 'blue', label = 'Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()




        ## modelo AdaBoost e decision tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor

regr_1 = DecisionTreeRegressor(max_depth=4)
rng = np.random.RandomState(1)
regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
                          n_estimators=300, random_state=rng)

regr_1.fit(x_train, y_train)
regr_2.fit(x_train, y_train)

# Predict
y_1 = regr_1.predict(x_train)
y_2 = regr_2.predict(x_train)

print('MAE:', mean_absolute_error(y_train, y_1))
print('MAE:', mean_absolute_error(y_train, y_2))

## graficos
plt.style.use("ggplot")
plt.figure(figsize=(12,8))
plt.xlabel('..')
plt.ylabel('Nota Matemática')
plt.title('Notas reais vs preditas')

plt.plot(y_train.index,y_pred_train)
plt.plot(y_train.index,y_train)

plt.legend(['Predições','Valores Reais'])
plt.show()


# histograma
        plt.hist(answer['NU_NOTA_MT'])
        x0=train_final['NU_NOTA_MT'].fillna(0)
        x1=answer['NU_NOTA_MT'].fillna(0)
        sns.distplot(x0)
        sns.distplot(x1)
        plt.legend(labels=['TRAIN','TEST'], ncol=2, loc='upper left')


## reposta
notas_pred = regressor.predict(x_test)
answer=pd.DataFrame()
answer['NU_INSCRICAO']=test['NU_INSCRICAO']
answer['NU_NOTA_MT'] = np.around(notas_pred,2) 
answer.to_csv('E:/Meu drive/desafio codenation/answer.csv', index=False, header=True)
