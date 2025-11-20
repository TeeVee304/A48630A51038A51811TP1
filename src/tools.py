import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

def get_best_params(param_grid, model, data, labels):
    """
    Obtém os melhores parâmetros para um dado modelo e conjunto de dados recorrendo ao GridSearchCV.
    Parâmetros
        param_grid : dict
            Dicionário com os nomes dos parâmetros (strings) como chaves e listas de valores a testar.
        model : Any
            Modelo a utilizar no GridSearchCV.
        data : numpy array
            Dados de entrada a utilizar no GridSearchCV.
        labels : numpy array
            Etiquetas correspondentes aos dados de entrada.
    Retorna
        dict : Dicionário com os melhores parâmetros encontrados pelo GridSearchCV.
    """
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid)
    grid_search.fit(data, labels)
    
    return grid_search.best_params_

def predition_stats(trueclass,prediction):
    """
    Calcula estatísticas de desempenho para um modelo de classificação.
    Parâmetros
        trueclass : numpy array
            Array com as classes verdadeiras.
        prediction : numpy array
            Array com as classes previstas.
    """
    cm = confusion_matrix(trueclass,prediction)
    
    true_pos = cm[0][0]
    false_neg = cm[0][1]
    true_neg = cm[1][1]
    false_pos = cm[1][0]
    
    total_positives = true_pos+false_neg
    total_negatives = true_neg + false_pos
    
    recall = np.round((true_pos / (total_positives))*100, 1)
    specificity = np.round((true_neg / total_negatives)*100, 1)
    pos_precision = np.round((true_pos / (true_pos+false_pos))*100, 1)
    neg_precision = np.round((true_neg / (true_neg+false_neg))*100, 1)
    
    fp_rate = np.round(false_pos/total_negatives,2)*100
    tn_rate = np.round(true_neg/total_negatives,2)*100
    tp_rate = np.round(true_pos/total_positives,2)*100
    fn_rate = np.round(false_neg/total_positives,2)*100
    
    F_score = np.round((2*pos_precision*recall)/(pos_precision + recall),4)
    G_score = np.round(np.sqrt(pos_precision*recall),4)
    
    print(f"\nTrue Positives {true_pos} ({tp_rate}%)\
            \nFalse Negatives {false_neg} ({fn_rate}%)\
            \n\nTrue Negatives {true_neg} ({tn_rate}%)\
            \nFalse Positives {false_pos} ({fp_rate}%)\
            \n\nRecall Rate: {recall}%\
            \nSpecificity Rate: {specificity}%\
            \n\nPositive Precision : {pos_precision}%\
            \nNegative Precision : {neg_precision}%\
            \n\nF-Score : {F_score}%\
            \nG-Score : {G_score}%")
    

def linearRegr(x,y,show_coef=False,show_regr_tax=False,plot_reta= False, plot_erros=False):
    """
    Função para fazer uma regressão linear com sklearn.
    Parâmetros
        x : numpy array
            Dados de entrada.
        y : numpy array
            Valores a prever.
        show_coef : bool     (False)
            Mostra os coeficientes da regressão.
        show_regr_tax : bool (False)
            Mostra o R2 da regressão.
        plot_reta : bool     (False)
            Plota a reta de regressão.
        plot_erros : bool    (False)
            Plota os erros da regressão.
    """
    x = np.array([x])
    y = np.array([y])
    
    linReg = LinearRegression().fit(x.T,y.T)
    
    w = linReg.coef_
    w0 =linReg.intercept_
    R2 = linReg.score(x.T,y.T)
    
    if show_coef:
        print(f'Coeficientes da regressão : {w0[0]}, {w}') 
    if show_regr_tax:
        print(f'R2 : {np.round(R2*100,2)}%')
        
def polinomialRegr(x,y,show_coef=False,show_regr_tax=False,plot_reta= False, plot_erros=False):
    """
    Função para fazer uma regressão polinomial de grau 3 com sklearn.
    """
    x = np.array([x])
    y = np.array([y])
    
    poly_3 = PolynomialFeatures(3).fit(x.T)
    # retiramos o termo independente colocado na linha anterior
    # pois este será internamente adicionado na regressão
    X_new = poly_3.transform(x.T)[:,1:]

    polyReg_3 = LinearRegression().fit(X_new,y.T)
    w=polyReg_3.coef_
    w0=polyReg_3.intercept_
    R2 = polyReg_3.score(X_new,y.T)
    #############################################################
    
    n_points = x.shape[1]
    X=np.vstack((np.ones((1,n_points)),x**3,x**2,x))

    Rx = np.dot(X,X.T) # X . XT
    rxy = np.dot(X,y.T) # X . yT
    
    w= np.dot(np.linalg.pinv(Rx),rxy) #(X . XT)^-1 . (X . yT)
    xmin = np.min(x)
    xmax = np.max(x)
    p1 = np.arange(xmin-1,xmax+1,0.25)
    n_points = p1.size
    p1_temp = np.vstack((np.ones(n_points),p1**3,p1**2,p1))
    p2 = np.dot(w.T,p1_temp)
    P= np.vstack((p1,p2))
    
    yh= np.dot(w.T,X) # calculo das saidas
    erros = y - yh
    
    if plot_reta:
        plt.plot(x[0,:],y[0,:],'x')
        plt.plot(P[0,:],P[1,:])
    
    if plot_erros:
        plt.plot(x,erros,'k.')
        plt.grid()
        plt.axis((xmin-1.,xmax+1.,np.min(erros)*3,np.max(erros)*3))
    
    if show_coef:
        print(f'\nW = {w0, w}')

    if show_regr_tax:
        print(f'R2 = {np.round(R2*100,2)}%')
        
def normalize_data(data):
    """
    Normaliza os dados de entrada para terem média 0 e variância 1.
    Parameters
        data : numpy array
            Dados de entrada a serem normalizados.
    Returns
        numpy array : Dados de entrada normalizados.
    """
    ss = StandardScaler().fit(data)
    return ss.transform(data)