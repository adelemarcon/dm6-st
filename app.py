# On importe les bibliothèques utiles
# -----------------------------------
import streamlit as st
from matplotlib import pyplot as plt
import numpy as np
from scipy.integrate import odeint

# Valeurs numériques caractéristiques du problème en unités SI
# ------------------------------------------------------------
q = -1.60e-19       # charge de l'électron (C)                 
m = 9.1e-31         # masse de l'électron (kg)   
v0 = 3e6            # vitesse initiale de l'électron (m/s)  
B0= 3.1             # valeur du champ magnétique (T)
Lambda = 9.1e-20    # coefficient de frottement fluide (kg/s)

k = st.slider('k: Gradient du champ magnétique', min_value=100000, max_value=2000000, step=1000)
# Paramètres de la résolution numérique
# ------------------------------------- 
t_fin = 60e-12                     # durée
N = 10000                          # nombre de points
t = np.linspace(0,t_fin,N)         # découpage du temps
W0 = [0,0,v0,0]                    # conditions initiales

# Définition du système différentiel d'ordre 1 à résoudre 
# ------------------------------------------------------- 
def systeme (W,t):
    Wp = np.zeros(4)                                                        # tableau vide pour stocker les dérivées
    Wp[0] = W[2]                                                            # première équation
    Wp[1] = W[3]                                                            # deuxième équation 
    Wp[2] = ((-q * B0) / m) * W[3] * (((((q * B0 * m) / Lambda**2) * W[3])-(m / Lambda) * W[2] +(m * v0) / Lambda) / (1 + (q * B0 /Lambda)**2) * k +1) - (Lambda / m) * W[2]      # troisième équation
    Wp[3] = ((q * B0) / m) * W[2] * (((((q * B0 * m) / Lambda**2) * W[3])-(m / Lambda) * W[2] +(m * v0) / Lambda) / (1 + (q * B0 /Lambda)**2) * k +1) - (Lambda /m) * W[3]       # quatrième équation 
    return Wp

# Intégration numérique à l'aide d'odeint
# ---------------------------------------
Wsol=odeint(systeme,W0,t)                        # la ligne qui fait le job ! 
x=Wsol[:,0]                                      # extraction de la position
y=Wsol[:,1]                                

# Représentation graphique 
# ------------------------
plt.plot(1e6*x,1e6*y)
plt.xlabel("x(µm)")
plt.ylabel("y(µm)")
plt.title("Document 1: Trajectoire de l'électron")
plt.grid()
plt.axis('equal')
st.pyplot(plt.gcf())