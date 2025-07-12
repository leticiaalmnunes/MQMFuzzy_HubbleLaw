#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Math

def cria_fuzzy_tri_eq (lista_x, meio_diametro):
    """Cria, a partir de uma lista ou dataframe de pontos e uma lista com seus respectivos diâmetros divididos por 2, números fuzzy triangulares equiláteros.
    Retorna os valores em uma lista de tuplas."""
    
    triplas = []
    for x, meio_diam in zip (lista_x, meio_diametro):
        triplas.append (tuple(((x-meio_diam), x, (x+meio_diam))))
    
    return triplas


def turn_fuzzy (df):
    """A partir de um conjunto de dados com valores reais, com uma variável independente 'x' e uma dependente 'y',
    transforma os valores de 'y' em números fuzzy triangulares equiláteros por meio de um ajuste linear clássico.
    
    Args:
        df: Tabela com os dados no formato de números reais. Pandas DataFrame.
    """
    cols = df.columns
    assert len(cols)==2
    
    X, y_true = df[cols[0]], df[cols[1]]
    
    # Ajuste linear
    a, b = np.polyfit(X, y_true, deg=1)
    y_prev = a*X +b
    
    # "Fuzzyficando"
    meio_diametro = [abs(y_t-y_p) for y_t, y_p in zip(y_true, y_prev)]
    Y = cria_fuzzy_tri_eq(y_true, meio_diametro)
    
    # Novo dataframe
    df_fuzzy = pd.DataFrame()
    df_fuzzy[0] = X
    df_fuzzy[1] = Y
    
    return df_fuzzy


def extrai_dataframe (df_pandas):
    """A partir de um dataframe pandas, cria a lista no formato adequado para que as funções próprias para números fuzzy triangulares sejam propriamente executadas.
    
    Args:
        Dataframe com duas colunas, sendo a primeira com os valores de x (números reais) e a segunda com os valores de y (números fuzzy triangulares - tuplas com 3 valores)."""
    
    colunas = df_pandas.columns
    assert len(colunas)==2
    
    df_novo = []
    for i in range(len(df_pandas)):
        df_novo.append( (df_pandas[colunas[0]][i], df_pandas[colunas[1]][i]) )
    
    return df_novo


def diametro (A):
    """Calcula o diâmetro de um número fuzzy triangular 'A'. 'A' deve ser uma tupla."""
    return (abs(A[2]-A[0]))

def soma0_Ft (A,B):
    """Calcula a soma zero (+_0) de dois números fuzzy triangulares"""
    p1=A[0]+B[2] # a + f
    p2=A[1]+B[1] # b + e
    p3=A[2]+B[0] # c + d
    diamA=abs(A[2]-A[0])
    diamB=abs(B[2]-B[0])

    if diamA >= diamB:
        s=(min(p1,p2), p2, max(p2,p3))
    else:
        s=(min(p3,p2), p2, max(p2,p1))
    
    if s[0]==s[2]:
        s=s[1]
    
    return s


def mult_Ft_e (F,e):
    """Multiplica um número fuzzy por um escalar real."""
    return tuple(np.array(F)*e)


def gera_A (df):
    """Gera uma matriz A 2x2, sendo Ax=b."""
    
    def g1(df):
        """g1(x)=x"""
        g1=[coord[0] for coord in df]    
        return np.array(g1)

    def g2(df):
        """g2(x)=x²"""
        g2=[coord[0] for coord in df]

        return np.array(g2)**2   
    
    A=np.zeros((2,2))
    A[0,0]=sum(g2(df))
    A[0,1]=A[1,0]=sum(g1(df))
    A[1,1]=len(df)
    
    return A


def gera_b (df):    
    """Gerando os dois valores da matriz 'b', sendo Ax=b, de forma separada. V1=(r,s,t) e V2=(u,v,w)."""
    df_novo=[]
    for coord in df: 
        novo_y=mult_Ft_e (coord[1],coord[0]) # multiplicando os valores de x e y (x_m*y_m)
        df_novo.append((coord[0],novo_y))
    
    V1=df_novo[0][1]
    V2=df[0][1]
    
    c=1
    while c<len(df):
        V1=soma0_Ft(V1,df_novo[c][1])
        V2=soma0_Ft (V2,df[c][1])
        c+=1
    
    return V1,V2


def sep_Vs(V1,V2):
    """Achando 'r', 's', 't', 'u', 'v' e 'w'. V1=(r,s,t) e V2=(u,v,w)."""
    return V1[0],V1[1],V1[2],V2[0],V2[1],V2[2]


def pivoteamento(A,b):
    """Realiza o pivoteamento parcial - ou seja, ordena as linhas - da matriz 'A' e, por concatenação,
    da matriz 'b' simultâneamente."""
    A = A.copy()
    b = b.copy()
    m,n = A.shape
    C = np.hstack((A,b)) # Concatenando 'A' e 'b'
    
    p = 0
    while p<n-1:
        M=max(abs(C[p:,p]))
        L=np.where(C[:,p]==M)[0][0] # achando o índice da linha do máximo absoluto
        
        if L>p:
            linha=C[L].copy() # trocando as linhas
            C[L]=C[p]
            C[p]=linha
        p += 1
    
    A, b = np.hsplit(C,[n]) # separando novamente as matrizes 'A' e 'b'
    
    return A, b


def LU (A,b):
    """Resolve por decomposição em LU um sistema do tipo Ax=b, retornando 'x'.
    Para evitar erros, pelo menos um elemento de cada matriz precisa ser um float."""
    n=A.shape[0]
    L=np.identity(n)
    U,b=pivoteamento(A,b)
    p=0

    if np.linalg.det(U)==0:
        return print('Esse sistema não pode ser resolvido por fatoração LU.')
    
    # criando as matrizes L e U
    while p<n:
        for i in range(p+1,n):
            m=U[i,p]/U[p,p] # achando os multiplicadores
            L[i,p]=m

            for j in range(p,n):                    
                U[i,j] = U[i,j]-(m*U[p,j]) # eliminação gaussiana para encontrar 'U'
        p+=1 

    y=np.dot((np.linalg.inv(L)),b) # y=Ux, ou seja, Ly=b.
    x=np.dot((np.linalg.inv(U)),y) # Ux=y
            
    return x



def condicoes1(M):
    S1= ( (M[0,0]>=0) and (M[0,1]>=0) and (M[1,0]>=0) and (M[1,1]>=0) )
    
    S2= ( (M[0,0]<=0) and (M[0,1]<=0) and (M[1,0]<=0) and (M[1,1]<=0) )
    
    S3= ( (M[0,0]>=0) and (M[0,1]>=0) and (M[1,0]<=0) and (M[1,1]<=0) )
    
    S4= ( (M[0,0]<=0) and (M[0,1]<=0) and (M[1,0]>=0) and (M[1,1]>=0) )
    
    return (S1 or S2 or S3 or S4) # retorna True (verdadeiro) ou False (falso)

def condicoes2(M):
    S1= ( (M[0,0]>=0) and (M[1,0]>=0) and (M[0,1]<=0) and (M[1,1]<=0) )
    
    S2= ( (M[0,0]>=0) and (M[1,1]>=0) and (M[0,1]<=0) and (M[1,0]<=0) )
    
    S3= ( (M[0,0]<=0) and (M[1,1]<=0) and (M[0,1]>=0) and (M[1,0]>=0) )
    
    S4= ( (M[0,0]<=0) and (M[1,0]<=0) and (M[0,1]>=0) and (M[1,1]>=0) )
    
    return (S1 or S2 or S3 or S4)

def condicoes3(M):
    S1= ( (M[0,0]>=0) and (M[0,1]>=0) and (M[1,0] >=0) and (M[1,1]<=0) )
    
    S2= ( (M[0,0]>=0) and (M[0,1]>=0) and (M[1,1] >=0) and (M[1,0]<=0) )
    
    S3= ( (M[0,0]<=0) and (M[0,1]<=0) and (M[1,1] <=0) and (M[1,0]>=0) )
    
    S4= ( (M[0,0]<=0) and (M[0,1]<=0) and (M[1,0] <=0) and (M[1,1]>=0) )
    
    return (S1 or S2 or S3 or S4)

def condicoes4(M):
    S1= ( (M[0,0]>=0) and (M[1,0]>=0) and (M[1,1]>=0) and (M[0,1]<=0) )
    
    S2= ( (M[0,1]>=0) and (M[1,0]>=0) and (M[1,1]>=0) and (M[0,0]<=0) )
        
    S3= ( (M[0,0]<=0) and (M[1,0]<=0) and (M[1,1] <=0) and (M[0,1]>=0) )
    
    S4= ( (M[0,1]<=0) and (M[1,0]<=0) and (M[1,1] <=0) and (M[0,0]>=0) )
    
    return (S1 or S2 or S3 or S4)


def metodo1 (M,x,V1,V2):
    Sol=[] # lista de possíveis soluções
    b=x[0,0]
    e=x[1,0]
    r,s,t,u,v,w=sep_Vs(V1,V2)
    
# 1º sistema
    Vp=np.array([[r],[u]])
    A = LU(M, Vp)[0,0]
    F = LU(M, Vp)[1,0]

    Vp=np.array([[t],[w]])
    C = LU(M, Vp)[0,0]
    D = LU(M, Vp)[1,0]

    if A<=b and b<=C and D<=e and e<=F:
        solucao=[(A,b,C),(D,e,F)]
        if solucao not in Sol: # evita soluções repetidas
            Sol.append(solucao)

# 2º sistema
    Vp=np.array([[t],[u]])
    C = LU(M, Vp)[0,0]
    D = LU(M, Vp)[1,0]

    Vp=np.array([[r],[w]])
    A = LU(M, Vp)[0,0]
    F = LU(M, Vp)[1,0]

    if A<=b and b<=C and D<=e and e<=F:
        solucao=[(A,b,C),(D,e,F)]
        if solucao not in Sol:
            Sol.append(solucao)
            
# 3º sistema
    Vp=np.array([[r],[u]])
    C = LU(M, Vp)[0,0]
    D = LU(M, Vp)[1,0]

    Vp=np.array([[t],[w]])
    A = LU(M, Vp)[0,0]
    F = LU(M, Vp)[1,0]

    if A<=b and b<=C and D<=e and e<=F:
        solucao=[(A,b,C),(D,e,F)]
        if solucao not in Sol:
            Sol.append(solucao)

# 4º sistema
    Vp=np.array([[r],[u]])
    C = LU(M, Vp)[0,0]
    D = LU(M, Vp)[1,0]

    Vp=np.array([[t],[w]])
    A = LU(M, Vp)[0,0]
    F = LU(M, Vp)[1,0]

    if A<=b and b<=C and D<=e and e<=F:
        solucao=[(A,b,C),(D,e,F)]
        if solucao not in Sol:
            Sol.append(solucao)
        
    return Sol



def metodo2(M,x,V1,V2):
    Sol=[] # lista de possíveis soluções
    b=x[0,0]
    e=x[1,0]
    r,s,t,u,v,w=sep_Vs(V1,V2)
    
# 1º sistema
    Vp=np.array([[r],[u]])
    A = LU(M, Vp)[0,0]
    D = LU(M, Vp)[1,0]

    Vp=np.array([[t],[w]])
    C = LU(M, Vp)[0,0]
    F = LU(M, Vp)[1,0]

    if A<=b and b<=C and D<=e and e<=F:
        solucao=[(A,b,C),(D,e,F)]
        if solucao not in Sol:
            Sol.append(solucao)

# 2º sistema
    Vp=np.array([[r],[w]])
    A = LU(M, Vp)[0,0]
    D = LU(M, Vp)[1,0]

    Vp=np.array([[t],[u]])
    C = LU(M, Vp)[0,0]
    F = LU(M, Vp)[1,0]

    if A<=b and b<=C and D<=e and e<=F:
        solucao=[(A,b,C),(D,e,F)]
        if solucao not in Sol:
            Sol.append(solucao)

# 3º sistema
    Vp=np.array([[r],[w]])
    C = LU(M, Vp)[0,0]
    F = LU(M, Vp)[1,0]

    Vp=np.array([[t],[u]])
    A = LU(M, Vp)[0,0]
    D = LU(M, Vp)[1,0]

    if A<=b and b<=C and D<=e and e<=F:
        solucao=[(A,b,C),(D,e,F)]
        if solucao not in Sol:
            Sol.append(solucao)

# 4º sistema
    Vp=np.array([[r],[u]])
    C = LU(M, Vp)[0,0]
    F = LU(M, Vp)[1,0]

    Vp=np.array([[t],[w]])
    A = LU(M, Vp)[0,0]
    D = LU(M, Vp)[1,0]

    if A<=b and b<=C and D<=e and e<=F:
        solucao=[(A,b,C),(D,e,F)]
        if solucao not in Sol:
            Sol.append(solucao)
        
    return Sol


def metodo3(M,x,V1,V2):
    Sol=[] # lista de possíveis soluções
    b=x[0,0]
    e=x[1,0]
    r,s,t,u,v,w=sep_Vs(V1,V2)
    
    z=np.array([[r],[u],[t],[w]])
    
    # criando as matrizes Z
    Z1=np.zeros((4,4))
    Z2=np.zeros((4,4))
    Z3=np.zeros((4,4))
    Z4=np.zeros((4,4))
    
    Z1[0,0],Z1[1,0] = M[0,0],M[1,0]
    Z1[2,1],Z1[3,1] = M[0,0],M[1,0]
    Z1[1,2],Z1[2,2] = M[1,1],M[0,1]
    Z1[0,3],Z1[3,3] = M[0,1],M[1,1]
    
    Z2[0,0],Z2[3,0] = M[0,0],M[1,0]
    Z2[1,1],Z2[2,1] = M[1,0],M[0,0]
    Z2[2,2],Z2[3,2] = M[0,1],M[1,1]
    Z2[0,3],Z2[1,3] = M[0,1],M[1,1]
    
    Z3[1,0],Z3[2,0] = M[1,0],M[0,0]
    Z3[0,1],Z3[3,1] = M[0,0],M[1,0]
    Z3[0,2],Z3[1,2] = M[0,1],M[1,1]
    Z3[2,3],Z3[3,3] = M[0,1],M[1,1]
    
    Z4[2,0],Z4[3,0] = M[0,0],M[1,0]
    Z4[0,1],Z4[1,1] = M[0,0],M[1,0]
    Z4[0,2],Z4[3,2] = M[0,1],M[1,1]
    Z4[1,3],Z4[2,3] = M[1,1],M[0,1]

    
    # 1º sistema
    A = LU(Z1, z)[0,0]
    C = LU(Z1, z)[1,0]
    D = LU(Z1, z)[2,0]
    F = LU(Z1, z)[3,0]

    if A<=b and b<=C and D<=e and e<=F:
        solucao=[(A,b,C),(D,e,F)]
        if solucao not in Sol:
            Sol.append(solucao)

    # 2º sistema
    A = LU(Z2, z)[0,0]
    C = LU(Z2, z)[1,0]
    D = LU(Z2, z)[2,0]
    F = LU(Z2, z)[3,0]

    if A<=b and b<=C and D<=e and e<=F:
        solucao=[(A,b,C),(D,e,F)]
        if solucao not in Sol:
            Sol.append(solucao)

    # 3º sistema
    A = LU(Z3, z)[0,0]
    C = LU(Z3, z)[1,0]
    D = LU(Z3, z)[2,0]
    F = LU(Z3, z)[3,0]

    if A<=b and b<=C and D<=e and e<=F:
        solucao=[(A,b,C),(D,e,F)]
        if solucao not in Sol:
            Sol.append(solucao)

    # 4º sistema
    A = LU(Z4, z)[0,0]
    C = LU(Z4, z)[1,0]
    D = LU(Z4, z)[2,0]
    F = LU(Z4, z)[3,0]

    if A<=b and b<=C and D<=e and e<=F:
        solucao=[(A,b,C),(D,e,F)]
        if solucao not in Sol:
            Sol.append(solucao)
        
    return Sol


def metodo4(M,x,V1,V2):
    Sol=[] # lista de possíveis soluções
    b=x[0,0]
    e=x[1,0]
    r,s,t,u,v,w=sep_Vs(V1,V2)
    
    z=np.array([[r],[u],[t],[w]])
    
    # criando as matrizes Z
    Z1=np.zeros((4,4))
    Z2=np.zeros((4,4))
    Z3=np.zeros((4,4))
    Z4=np.zeros((4,4))
    
    Z1[1,0],Z1[2,0] = M[1,0],M[0,0]
    Z1[0,1],Z1[3,1] = M[0,0],M[1,0]
    Z1[2,2],Z1[3,2] = M[0,1],M[1,1]
    Z1[0,3],Z1[1,3] = M[0,1],M[1,1]
    
    Z2[2,0],Z2[3,0] = M[0,0],M[1,0]
    Z2[0,1],Z2[1,1] = M[0,0],M[1,0]
    Z2[1,2],Z2[2,2] = M[1,1],M[0,1]
    Z2[0,3],Z2[3,3] = M[0,1],M[1,1]
    
    Z3[0,0],Z3[1,0] = M[0,0],M[1,0]
    Z3[2,1],Z3[3,1] = M[0,0],M[1,0]
    Z3[0,2],Z3[3,2] = M[0,1],M[1,1]
    Z3[1,3],Z3[2,3] = M[1,1],M[0,1]
    
    Z4[0,0],Z4[3,0] = M[0,0],M[1,0]
    Z4[1,1],Z4[2,1] = M[1,0],M[0,0]
    Z4[0,2],Z4[1,2] = M[0,1],M[1,1]
    Z4[2,3],Z4[3,3] = M[0,1],M[1,1]

    
    # 1º sistema
    A = LU(Z1, z)[0,0]
    C = LU(Z1, z)[1,0]
    D = LU(Z1, z)[2,0]
    F = LU(Z1, z)[3,0]

    if A<=b and b<=C and D<=e and e<=F:
        solucao=[(A,b,C),(D,e,F)]
        if solucao not in Sol:
            Sol.append(solucao)

    # 2º sistema
    A = LU(Z2, z)[0,0]
    C = LU(Z2, z)[1,0]
    D = LU(Z2, z)[2,0]
    F = LU(Z2, z)[3,0]

    if A<=b and b<=C and D<=e and e<=F:
        solucao=[(A,b,C),(D,e,F)]
        if solucao not in Sol:
            Sol.append(solucao)

    # 3º sistema
    A = LU(Z3, z)[0,0]
    C = LU(Z3, z)[1,0]
    D = LU(Z3, z)[2,0]
    F = LU(Z3, z)[3,0]

    if A<=b and b<=C and D<=e and e<=F:
        solucao=[(A,b,C),(D,e,F)]
        if solucao not in Sol:
            Sol.append(solucao)

    # 4º sistema
    A = LU(Z4, z)[0,0]
    C = LU(Z4, z)[1,0]
    D = LU(Z4, z)[2,0]
    F = LU(Z4, z)[3,0]

    if A<=b and b<=C and D<=e and e<=F:
        solucao=[(A,b,C),(D,e,F)]
        if solucao not in Sol:
            Sol.append(solucao)
        
    return Sol


def ACDF (M,x,V1,V2):
    """Define em qual situação a matriz 2x2 M se encaixa para que o método de resolução seja escolhido
    e retorna as possíveis soluções do sistema."""
    if (condicoes1(M)):
        Sol=metodo1(M,x,V1,V2)
    elif (condicoes2(M)):
        Sol=metodo2(M,x,V1,V2)
    elif (condicoes3(M)):
        Sol=metodo3(M,x,V1,V2)
    elif (condicoes4(M)):
        Sol=metodo4(M,x,V1,V2)

    return Sol


def sistema_fuzzy (df):
    """Monta um sistema de equações lineares fuzzy a partir de um conjunto de dados e o resolve.
    Apenas para matrizes 2x2 que podem ser resolvidas por ajuste linear."""
    
    # MÍNIMOS QUADRADOS
    A = gera_A(df)
    V1, V2 = gera_b(df)

    r, s, t, u, v, w = sep_Vs(V1,V2)
    Vp = np.array([[s],[v]])
    x = LU(A,Vp)   
    
    solução = ACDF(A,x,V1,V2)
    
    return solução

########################################################################################################################################

def exibe_linfunc_fuzzy (coeficientes):
    for i, sol in enumerate(coeficientes):
        # Convertendo as tuplas para string
        a_str = ",\ ".join([f"{coef:.2f}" for coef in sol[0]])
        b_str = ",\ ".join([f"{coef:.2f}" for coef in sol[1]])

        expressao = fr"y_{i+1} = ({a_str})x +_{{0}} ({b_str})"
        display(Math(expressao))
        

def plota_ajuste_fuzzy_2D(df, titulo=None, nome_x = None, nome_y = None, label_dado = None, ticks_x=[], sep_solutions = False,
                       dimension = [10,6], quality = 100, save_fig = False, display_function = False):
    """
    Exibe o gráfico do ajuste linear fuzzy por quadrados mínimos a partir de um conjunto de dados.
    
    Args:
        df: Conjunto de dados fuzzy no formato de lista de tuplas, sendo cada tupla uma coordenada (x,y),
        com o y sendo também uma tupla que representa um número fuzzy triangular.
        
        titulo: Título que será exibido no gráfico e, caso 'save_fig==True', na figura salva. String. Valor padrão: None.
        
        nome_x: Nome dado ao eixo x. String. Valor padrão: None.
        
        nome_y: Nome dado ao eixo y. String. Valor padrão: None.
        
        label_dado: Nome dado aos dados na legenda. String. Valor padrão: None.
        
        ticks_x: Altera os valores dos dados que são exibidos no eixo x.
        Recomendado apenas para datasets pequenos cujos valores de x estão igualmente espaçados.
        Caso não seja fornecido, será mantido o padrão do matplotlib. Lista. Valor padrão: [].
        
        sep_solutions: Argumento que indica se as soluções devem ser exibidas em gráficos separados.
        Booleano. Valor padrão: False.
        
        dimension: Dimensão do gráfico (2 dimensões). Lista. Valor padrão: [10,6].
        
        quality: Qualidade da imagem (dpi). Inteiro. Valor padrão: 100.
        
        save_fig: Argumento que indica se devem ser geradas imagens (png) para os gráficos obtidos.
        As imagens serão salvas no mesmo diretório no qual a função for utilizada. Booleano. Valor padrão: False.
        
        display_function: Argumento que indica se as equações das soluções devem ser exibidas. Booleano. Valor padrão: False.
    """
    
    solucoes = sistema_fuzzy (df)        
    xs = np.arange(df[0][0],df[-1][0]+0.125,0.125)
    c = 0
    cores = ['#5f2ad5','#5271ff','blue','darkgreen']
    
    def sep_y():
        """Separa as componentes de y."""
        ys=[]
        for x in xs:
            ax = mult_Ft_e (C1,x)
            ys.append(soma0_Ft(ax,C2))

        y0=np.array([y[0] for y in ys])
        y1=np.array([y[1] for y in ys])
        y2=np.array([y[2] for y in ys])

        return y0, y1, y2

    
    def superficie():
        """Plota a suferfície do ajuste no gráfico."""
        plt.plot(xs,y0,color=cores[c]) # Inf

        if c==0 or sep_solutions:
            plt.plot(xs,y1,color='#000000', label='Pico')
        else:
            plt.plot(xs,y1,color='#000000') # Pic

        plt.plot(xs,y2,color=cores[c], label=f'{c+1}º ajuste obtido') # Sup

        plt.fill_between(xs,y0.flatten(),y2.flatten(),color=cores[c],alpha=0.10)


    def dados():
        """Plota os pontos no gráfico."""
        X=[coord[0] for coord in df]
        Y=[coord[1][1] for coord in df]
        plt.scatter(X,Y,color='#333333', marker='o', label = label_dado) # pontos
        
        p=0
        for coord in df:
            m=[coord[0],coord[0]]
            n=[coord[1][0],coord[1][2]]
            plt.plot(m,n,color='#777777', label=('Incerteza' if p==0 else None), alpha=.7) # barras
            p+=1

    def info(titulo_graf):
        """Formatação das informações exibidas no gráfico."""
        plt.title(titulo_graf, fontsize='x-large')

        if len(ticks_x)!= 0:
            ticks0_x = [df[i][0] for i in range (len(df))]
            plt.xticks(ticks0_x,ticks_x)
        else:
            pass

        plt.xlim(0,500) # APAGAR
        plt.ylim(0,35000)
        
        plt.xlabel(nome_x, fontsize='large')
        plt.ylabel(nome_y, fontsize='large')
        plt.legend(loc='best', fontsize='medium')
    
    # GERANDO O(S) GRÁFICO(S)
    if sep_solutions:
        for sol in range(len(solucoes)):
            plt.figure(figsize=dimension, dpi=quality)
            titulo_graf = titulo + f' - Solução {c+1}'
            
            C1, C2 = solucoes[sol]            
            y0, y1, y2 = sep_y()
            superficie()
            dados()
            info(titulo_graf)

            if save_fig:
                plt.savefig(f'Graph_{titulo_graf}')

            c+=1

    else:
        plt.figure(figsize=dimension, dpi=quality)

        for sol in range(len(solucoes)):
            C1, C2 = solucoes[sol]
            y0, y1, y2 = sep_y()
            superficie()
            c+=1

        dados()
        info(titulo)
        
        if save_fig:
            plt.savefig(f'Graph_{titulo}')
    
    if display_function:
        exibe_linfunc_fuzzy (solucoes)