# Import
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import base64
import os
import altair as alt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from streamlit_folium import folium_static
from sklearn.model_selection import learning_curve
import folium

# Create a title and a subtitle
st.title("Plus de transparence sur le marché de l'occasion utilitaire benne")
st.write(
    "A partir d'une base de donnees de plus de 1000 annonces nous avons developpé un algorithme d'intelligence artificielle capable de vous proposer le prix auquel des vehicules similaires au vôtre sont vendus ainsi que les délais selon votre prix rêvé !")

# Open and display an image
image = Image.open('Logo.png')
st.sidebar.image(image, use_column_width=True)

# get the data
df = pd.read_csv('/Users/mackookproyann/PycharmProjects/pythonProject/012021_bdd occasions - BdD.csv')
df1 = pd.read_csv('/Users/mackookproyann/PycharmProjects/pythonProject/012021_bdd occasions - BdD S1.csv')
df2 = pd.read_csv('/Users/mackookproyann/PycharmProjects/pythonProject/012021_bdd occasions - BdD S2.csv')
df3 = pd.read_csv('/Users/mackookproyann/PycharmProjects/pythonProject/012021_bdd occasions - BdD S4.csv')
df4 = pd.read_csv('/Users/mackookproyann/PycharmProjects/pythonProject/012021_bdd occasions - BdD_plot.csv')
annonces = pd.read_csv(
    '/Users/mackookproyann/PycharmProjects/pythonProject/012021_bdd occasions - Nbr annonces_département.csv')
geo_vendu = pd.read_csv(
    '/Users/mackookproyann/PycharmProjects/pythonProject/012021_bdd occasions - %S4_département.csv')
corr_prix = pd.read_csv(
    '/Users/mackookproyann/PycharmProjects/pythonProject/012021_bdd occasions - correlation_prix.csv')

# Split the data into independent 'X' and dependent 'Y' variables
Y = df['Prix (TTC)']
X = df.drop(['Prix (TTC)', 'Date', 'Département', 'Marque', 'Cabine',
             'Boîte vitesse', 'Carburant', 'Puissance réelle (ch)', 'Carrosserie',
             'Coffre', 'Réhausse',
             'Mention benne JPM', 'Garantie (mois)', 'Lien', 'S1 vendu', 'S2 vendu', 'S4 vendu', 'Modification Prix',
             'Prix initial', 'Date installation benne'], axis=1)
# Split the data set into 75% training and 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

numerical_features_prix = ['Annee', 'Km']
categorical_features_prix = ['Modele']
numerical_pipeline_prix = make_pipeline(SimpleImputer(),
                                        StandardScaler())
categorical_pipeline_prix = make_pipeline(SimpleImputer(strategy='most_frequent'),
                                          OneHotEncoder())
preprocessor_prix = make_column_transformer((numerical_pipeline_prix, numerical_features_prix),
                                            (categorical_pipeline_prix, categorical_features_prix))


V = df1['S1 vendu']
U = df1.drop(['S1 vendu', 'Date', 'Département', 'Marque', 'Cabine',
             'Boîte vitesse', 'Carburant', 'Puissance réelle (ch)', 'Carrosserie',
             'Coffre', 'Réhausse',
             'Mention benne JPM', 'Garantie (mois)', 'Lien', 'S2 vendu', 'S4 vendu', 'Modification Prix',
             'Prix initial', 'Date installation benne'], axis=1)
U_train, U_test, V_train, V_test = train_test_split(U, V, test_size=0.25, random_state=0)

numerical_features = ['Annee', 'Km', 'Prix (TTC)']
categorical_features = ['Modele']
numerical_pipeline = make_pipeline(SimpleImputer(strategy='mean'),
                                   StandardScaler())
categorical_pipeline = make_pipeline(SimpleImputer(strategy='most_frequent'),
                                     OneHotEncoder())
preprocessor = make_column_transformer((numerical_pipeline, numerical_features),
                                       (categorical_pipeline, categorical_features))


N = df2['S2 vendu']
M = df2.drop(['S2 vendu', 'Date', 'Département', 'Marque', 'Cabine',
             'Boîte vitesse', 'Carburant', 'Puissance réelle (ch)', 'Carrosserie',
             'Coffre', 'Réhausse',
             'Mention benne JPM', 'Garantie (mois)', 'Lien', 'S1 vendu', 'S4 vendu', 'Modification Prix',
             'Prix initial', 'Date installation benne'], axis=1)
M_train, M_test, N_train, N_test = train_test_split(M, N, test_size=0.25, random_state=0)

numerical_features_S2 = ['Annee', 'Km', 'Prix (TTC)']
categorical_features_S2 = ['Modele']
numerical_pipeline_S2 = make_pipeline(SimpleImputer(strategy='mean'),
                                   StandardScaler())
categorical_pipeline_S2 = make_pipeline(SimpleImputer(strategy='most_frequent'),
                                     OneHotEncoder())
preprocessor_S2 = make_column_transformer((numerical_pipeline_S2, numerical_features_S2),
                                       (categorical_pipeline_S2, categorical_features_S2))

#O = df3.iloc[:, 0:4].values
#P = df3.iloc[:, -1].values
P = df3['S4 vendu']
O = df3.drop(['S4 vendu', 'Date', 'Département', 'Marque', 'Cabine',
             'Boîte vitesse', 'Carburant', 'Puissance réelle (ch)', 'Carrosserie',
             'Coffre', 'Réhausse',
             'Mention benne JPM', 'Garantie (mois)', 'Lien', 'S1 vendu', 'S2 vendu', 'Modification Prix',
             'Prix initial', 'Date installation benne'], axis=1)
O_train, O_test, P_train, P_test = train_test_split(O, P, test_size=0.25, random_state=0)

numerical_features_S4 = ['Annee', 'Km', 'Prix (TTC)']
categorical_features_S4 = ['Modele']
numerical_pipeline_S4 = make_pipeline(SimpleImputer(strategy='mean'),
                                   StandardScaler())
categorical_pipeline_S4 = make_pipeline(SimpleImputer(strategy='most_frequent'),
                                     OneHotEncoder())
preprocessor_S4 = make_column_transformer((numerical_pipeline_S4, numerical_features_S4),
                                       (categorical_pipeline_S4, categorical_features_S4))

# Default settings
st.sidebar.title("Estimation du prix")


# Get the feature from the user
def get_user_input():
    # initialize list of lists
    vehicule = st.sidebar.selectbox(label="Sélectionnez un modèle", options=['Daily', 'Master', 'Crafter',
                                                                             'Transit', 'Sprinter', 'Ducato', 'Jumper',
                                                                             'Canter'
                                                                             'Boxer', 'Cabstar', 'Maxity'])

    annee = st.sidebar.slider('Année', 1998, 2021, 2018)
    km = st.sidebar.text_input('Km', '75345')

    # Store a dictionary into a variable
    user_data = {'Modele': vehicule,
                 'Annee': annee,
                 'Km': km
                 }

    # Transform the data into a data frame
    features = pd.DataFrame(user_data, index=[0])

    return features


# Store the user input into a variable
user_input = get_user_input()

# Set a subheader and display the users input
st.header('Estimation du prix de vente')
st.write('Récapitulatif des paramètres rentrés:')
st.write(user_input)

# Create and train the model
model_prix = make_pipeline(preprocessor_prix, LinearRegression())
model_prix.fit(X_train, Y_train)

# Store the models predictions in a variable
prediction = model_prix.predict(user_input)

# Set a subheader display result
st.subheader('Proposition de prix et précision du modèle :')
accuracy_score = model_prix.score(X_test, Y_test)
st.write(str(int(prediction)) + ' €. Avec un score de précision de: ' + str(round(accuracy_score * 100)) + ' %')

#B, train_score, val_score = learning_curve(model_prix,X_train, Y_train, train_sizes = np.linspace(0.2, 1.0, 5), cv=5)
#st.write(B)
#plt.plot(B, train_score.mean(axis=1), label='train')
#plt.plot(B,val_score.mean(axis=1), label='validation')
#plt.xlabel('train_sizes')
#plt.legend()
#st.pyplot()

# Default settings
st.sidebar.title("Estimation du temps de vente")


# Get the feature from the user
def get_user_input2():
    vehicule2 = user_input['Modele']
    annee2 = user_input['Annee']
    km2 = user_input['Km']
    prix = st.sidebar.text_input('Entrez votre prix espéré (TTC)', '20000')

    # Store a dictionary into a variable
    user_data2 = {'Modele': vehicule2,
                  'Annee': annee2,
                  'Km': km2,
                  'Prix (TTC)': prix
                  }

    # Transform the data into a data frame
    features2 = pd.DataFrame(user_data2, index=[0])

    return features2


# Store the user input into a variable
user_input2 = get_user_input2()


# Get the feature from the user
def get_user_input3():
    vehicule3 = user_input['Modele']
    annee3 = user_input['Annee']
    km3 = user_input['Km']
    prix3 = user_input2['Prix (TTC)']

    # Store a dictionary into a variable
    user_data2 = {'Modele': vehicule3,
                  'Annee': annee3,
                  'Km': km3,
                  'Prix (TTC)': prix3
                  }

    # Transform the data into a data frame
    features3 = pd.DataFrame(user_data2, index=[0])

    return features3


# Store the user input into a variable
user_input3 = get_user_input3()


# Get the feature from the user
def get_user_input4():
    vehicule4 = user_input['Modele']
    annee4 = user_input['Annee']
    km4 = user_input['Km']
    prix4 = user_input2['Prix (TTC)']

    # Store a dictionary into a variable
    user_data2 = {'Modele': vehicule4,
                  'Annee': annee4,
                  'Km': km4,
                  'Prix (TTC)': prix4
                  }

    # Transform the data into a data frame
    features4 = pd.DataFrame(user_data2, index=[0])

    return features4


# Store the user input into a variable
user_input4 = get_user_input4()

# Set a subheader and display the users input
st.header('')
st.header('Estimation du temps de vente')
st.write('Récapitulatif des paramètres pour estimation temps de vente:')
st.write(user_input2)

# Create & train the model
model = make_pipeline(preprocessor, SGDClassifier())
model.fit(U_train, V_train)

prediction2 = model.predict(user_input2)
accuracy_score2 = model.score(U_test, V_test)

model_S2 = make_pipeline(preprocessor_S2, SGDClassifier())
model_S2.fit(M_train,N_train)

prediction3 = model_S2.predict(user_input3)
accuracy_score3 = model_S2.score(M_test, N_test)

model_S4 = make_pipeline(preprocessor_S4, SGDClassifier())
model_S4.fit(O_train,P_train)

prediction4 = model_S4.predict(user_input3)
accuracy_score4 = model_S4.score(O_test, P_test)


# Set a subheader and display classification
st.subheader('Délai de vente et score de précision du modèle: ')
if prediction2 == "Non":
    prediction2_text = '**Non vendu**'
else:
    prediction2_text = '**Vendu**'
if prediction3 == "Non":
    prediction3_text = '**Non vendu**'
else:
    prediction3_text = '**Vendu**'
if prediction4 == "Non":
    prediction4_text = '**Non vendu**'
else:
    prediction4_text = '**Vendu**'
st.write('Semaine 1 : ' + str(prediction2_text) + '. Avec un score de précision de: ' + str(
    round(accuracy_score2 * 100)) + '%')

st.write('Semaine 2 : ' + str(prediction3_text)+'. Avec un score de précision de: '+ str(round(accuracy_score3 * 100))+ '%')
st.write('Semaine 4 : ' + str(prediction4_text)+'. Avec un score de précision de: '+ str(round(accuracy_score4 * 100))+ '%')



### gif from local file
@st.cache
def get_base64_of_bin_file(bin_file):
    with open("/Users/mackookproyann/PycharmProjects/pythonProject/ gif_options_V3.gif", 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


@st.cache
def get_img_with_href(local_img_path, target_url):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}">
            <img src="data:image/{img_format};base64,{bin_str}" />
        </a>'''
    return html_code


gif_html = get_img_with_href('tenor.gif', 'https://www.jpm-group.com/fr/option-1')
st.sidebar.markdown(gif_html, unsafe_allow_html=True)

# Set a subheader
st.header('')
st.header('Jouez avec les données:')
st.write("Notre modèle est bâti grâce à l'analyse de plus de 1000 annonces de véhicules d'occasions.")

st.subheader("Les nombres d'annonces en fonction de differents paramètres")
selected_metrics = st.selectbox(label="Choisissez le paramètre",
                                options=["Nombre d'annonces par modèle","Nombre d'annonces par année", "Nombre d'annonces par kilométrage", "Nombre d'annonces par puissance","Nombre d'annonces par fourchettes de prix",
                                         "Les marques de benne les plus précisées dans les annonces"])
if selected_metrics == "Nombre d'annonces par modèle":
    st.bar_chart(df4['Modele'].value_counts())
if selected_metrics == "Nombre d'annonces par année":
    st.bar_chart(df4['Annee'].value_counts())
if selected_metrics == "Nombre d'annonces par kilométrage":
    st.bar_chart(df4['Km'].value_counts())
if selected_metrics == "Nombre d'annonces par puissance":
    st.bar_chart(df4['Puissance réelle (ch)'].value_counts())
if selected_metrics == "Nombre d'annonces par fourchettes de prix":
    st.bar_chart(df4['Prix (TTC)'].value_counts())
if selected_metrics == "Les marques de benne les plus précisées dans les annonces":
    st.bar_chart(df4['Marque Benne'].value_counts())

st.subheader("Quels types de véhicules sont vendus le plus rapidement ?")
selected_week = st.selectbox(label="Sélectionnez un délai", options=['1 semaine', '2 semaines', '4 semaines'])
if selected_week == '1 semaine':
    fig1 = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["Toutes annonces", "Vendus", "Non vendus", "5000€ / +200 000km", "10 000€ / 150 000km",
                   "10 000€ / +200 000km", "25 000€ / -50 000km", "20 000€ / 100 000km", "+30 000€ / -50 000km",
                   "25 000€ / -50 000km", "20 000€ / -50 000km"],
            color="#4b77a9"
        ),
        link=dict(
            source=[0, 0, 1, 1, 1, 1, 2, 2, 2, 2],  # indices correspond to labels, eg A1, A2, A1, B1, ...
            target=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            value=[262, 731, 39, 23, 34, 15, 58, 167, 109, 83]
        ))])

    fig1.update_layout(title_text="Véhicules vendus après 1 semaine (prix et km arrondis)", font_size=10)
    st.plotly_chart(fig1)

if selected_week == '2 semaines':
    fig2 = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["Toutes annonces", "Vendus", "Non vendus", "25 000€ / -50 000km", "+30 000€ / -50 000km",
                   "15 000€ / -50 000km", "5 000€ / +200 000km", "+30 000€ / -50 000km", "25 000€ / -50 000km",
                   "20 000€ / -50 000km", "20 000€ / 100 000km"],
            color="#4b77a9"
        ),
        link=dict(
            source=[0, 0, 1, 1, 1, 1, 2, 2, 2, 2],  # indices correspond to labels, eg A1, A2, A1, B1, ...
            target=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            value=[139, 266, 18, 14, 11, 71, 48, 31, 23]
        ))])
    fig2.update_layout(title_text="Véhicules vendus après 2 semaines (prix et km arrondis)", font_size=10)
    st.plotly_chart(fig2)

if selected_week == '4 semaines':
    fig4 = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=["Toutes annonces", "Vendus", "Non vendus", "25 000€ / -50 000km", "+30 000€ / -50 000km",
                   "15 000€ / -50 000km", "5 000€ / +200 000km", "+30 000€ / -50 000km", "25 000€ / -50 000km",
                   "20 000€ / -50 000km", "20 000€ / 100 000km"],
            color="#4b77a9"
        ),
        link=dict(
            source=[0, 0, 1, 1, 1, 1, 2, 2, 2, 2],  # indices correspond to labels, eg A1, A2, A1, B1, ...
            target=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            value=[87, 93, 15, 9, 8, 6, 16, 16, 13, 10]
        ))])

    fig4.update_layout(title_text="Véhicules vendus après 4 semaines (prix et km arrondis)", font_size=10)
    st.plotly_chart(fig4)


### gif from local file
@st.cache
def get_base64_of_bin_file(bin_file):
    with open("/Users/mackookproyann/PycharmProjects/pythonProject/img_garantie.png", 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


@st.cache
def get_img_with_href(local_img_path, target_url):
    img_format = os.path.splitext(local_img_path)[-1].replace('.', '')
    bin_str = get_base64_of_bin_file(local_img_path)
    html_code = f'''
        <a href="{target_url}">
            <img src="data:image/{img_format};base64,{bin_str}" />
        </a>'''
    return html_code


gif_html = get_img_with_href('tenor.gif', 'https://www.jpm-group.com/fr/garantie')
st.markdown(gif_html, unsafe_allow_html=True)

st.title('')
st.subheader('')

st.subheader("Quels départements sont-ils les plus dynamiques ?")
selected_geo_criteria = st.selectbox(label="Sélectionnez un critère", options=["Nombre d'annonces par département",
                                                                               "% de véhicules vendus après 4 semaines par département"])

if selected_geo_criteria == "Nombre d'annonces par département":
    m = folium.Map(location=[46.232193, 2.209667],
                   tiles='CartoDB positron',
                   zoom_start=5,
                   width='100',
                   height='60')


    def circle_maker(x):
        folium.Circle(location=[x[1], x[2]],
                      radius=float(x[3]) * 2000,
                      color='#4b77a9',
                      fill=True,
                      popup='{}. \n Nombre annonces: {}'.format(x[0], x[3])).add_to(m)


    annonces[['Departement', 'latitude', 'longitude', 'Nombre annonces']].apply(lambda x: circle_maker(x), axis=1)

    folium_static(m)

if selected_geo_criteria == "% de véhicules vendus après 4 semaines par département":
    m = folium.Map(location=[46.232193, 2.209667],
                   tiles='CartoDB positron',
                   zoom_start=5,
                   width='100',
                   height='60')


    def circle_maker(x):
        folium.Circle(location=[x[1], x[2]],
                      radius=float(x[4]) * 100000,
                      color='#4b77a9',
                      fill=True,
                      popup='{}. \n Vente à S4: {}'.format(x[0], x[3])).add_to(m)


    geo_vendu[['Departement', 'latitude', 'longitude', '%_vendu_S4', '%vente']].apply(lambda x: circle_maker(x), axis=1)

    folium_static(m)

st.subheader("Quels critères influent-ils le plus sur le prix ?")
st.write("Les corrélations se rangent de -1 à +1. Les values proches de 0 signifient qu'il n'y pas de relation "
         "linéaire entre le prix et la variable. Les plus proches de 1 sont corrélées positivement, c'est-à-dire que "
         "lorsque ces variables augmentent le prix fait de même. A l'inverse pour les variables en négatif, "
         "elles font baisser le prix. Bref, les valeurs proches de +1 ou -1 ont une forte influence sur le prix (à la "
         "hausse pour +1 et à la baisse pour -1)")
f, ax = plt.subplots(figsize=(5, 0.5))
cmap = sns.cubehelix_palette(start=.5, rot=-.5, as_cmap=True)
sns.heatmap(corr_prix.corr().loc[['Prix (TTC)'], :], ax=ax, cmap=cmap, yticklabels=False, annot=True, cbar=False)
plt.tick_params(axis='both', labelsize=5)
st.pyplot(f)

st.header('')
st.header("Pourquoi ce site ?")
st.write(
    "Sur les sites actuels soit les recherches sont limitées avec à chaque fois des conditions à accepter pour être spamé, soit il n'y a pas les cotes des VUL, soit les deux !")
st.write('Transférez à un ami à chaque fois que vous avez vécu une des situations ci-dessous : ')

video_file = open('/Users/mackookproyann/PycharmProjects/pythonProject/Mon film 7.mp4', 'rb')
video_bytes = video_file.read()
st.video(video_bytes)
