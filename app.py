from flask import Flask, render_template, request, redirect, url_for, session, flash, send_file, jsonify

# --- Système et fichiers ---
import os
import time
import re
from datetime import datetime, timedelta
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.service import Service

# --- Bases de données et données ---
import sqlite3
import pandas as pd
import numpy as np
import threading
import webview
import webbrowser

# --- Optimisation et calcul ---
import cvxpy as cp
from scipy.optimize import minimize

# --- Emails ---
from email.message import EmailMessage
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
from werkzeug.security import generate_password_hash, check_password_hash

# --- Selenium et Web scraping ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
from webdriver_manager.chrome import ChromeDriverManager

# --- Internet et API ---
import requests


# --- Visualisation ---
import plotly.express as px
import plotly.io as pio

# --- Utilitaires ---
import random
from apscheduler.schedulers.background import BackgroundScheduler
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import atexit
# --- Selenium helper ---
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

def make_chrome(headless=True):
    opts = Options()
    if headless:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-gpu")
    service = Service(ChromeDriverManager().install())
    return webdriver.Chrome(service=service, options=opts)
app = Flask(__name__)
DATABASE = r'C:\Users\HP\TRY\ma_base2.sqlite'
EXCEL_PATH = "matrices_correlation_covariance.xlsx"

def dict_factory(cursor, row):
    return {col[0]: row[idx] for idx, col in enumerate(cursor.description)}

conn = sqlite3.connect("base_actions1.db") 

conn.row_factory = dict_factory


def get_db_connection():
    if not os.path.exists(DATABASE):
        raise FileNotFoundError(f"La base de données SQLite n'existe pas : {DATABASE}")
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def get_table_name():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
    if not tables:
        raise Exception("Aucune table trouvée dans la base de données.")
    return tables[0][0]


@app.route('/resultats')
def resultats():
    nom = request.args.get('nom', '').strip()
    table = get_table_name()
    conn = get_db_connection()
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()

    if nom not in df['denomination_opcvm'].values:
        return render_template('resultat.html', nom=nom, resultat=[], message="Aucun OPCVM trouvé.")

    cluster = df.loc[df['denomination_opcvm'] == nom, 'cluster'].values[0]
    similaires = df[df['cluster'] == cluster]

    return render_template('resultat.html', nom=nom,
                           resultat=similaires.to_dict(orient='records'),
                           message=None)

@app.route('/recommandation', methods=['POST'])
def recommandation():
    data = request.get_json()
    profil = data.get('profil')
    nombreOpcvm = data.get('nombreOpcvm')

    # Simule le calcul du portefeuille optimal (exemple)
    # Ici tu peux appeler ta vraie fonction de calcul
    portefeuille_optimal = f"Portefeuille optimal calculé pour profil {profil} avec {nombreOpcvm} OPCVM."

    # Prépare un message complet
    message = f"Profil: {profil}, Nombre OPCVM: {nombreOpcvm}. {portefeuille_optimal}"

    return jsonify({'message': message})

@app.route('/recommander', methods=['POST'])
def recommander():
    nom = request.form['nom'].strip()
    return redirect(url_for('resultats', nom=nom))

@app.route('/recherche')
def recherche():
    return render_template('recherche.html')

@app.route('/clusterO')
def recherche1():
    return render_template('recherche.html')
@app.route('/aleatoire')
def aleatoire():
    return render_template('aleatoire.html')

def get_opcvm_data():
    """Charge les OPCVM depuis la base SQLite"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT libelle, Rendement_annuel_simple, Volatilite_annuelle FROM opcvm", conn)
    conn.close()
    return df


def markowitz_portfolio(df):
    """Optimisation de Markowitz : min variance avec contrainte somme des poids = 1"""
    n = len(df)
    mu = df["Rendement_annuel_simple"].values
    sigma = np.diag(df["Volatilite_annuelle"].values ** 2)  # si pas de corrélations dans ta base

    # Variables
    w = cp.Variable(n)

    # Problème : minimiser la variance
    risk = cp.quad_form(w, sigma)
    prob = cp.Problem(cp.Minimize(risk), [cp.sum(w) == 1, w >= 0])

    prob.solve()

    return w.value

@app.route("/generer_portefeuille", methods=["POST"])
def generer_portefeuille():
    data = request.json
    profil = data.get("profil")
    nombre = int(data.get("nombre", 5))

    # Charger OPCVM
    df = get_opcvm_data()

    # Tirer aléatoirement les OPCVM
    if nombre > len(df):
        nombre = len(df)
    choix = df.sample(n=nombre, random_state=None)

    # Optimiser avec Markowitz
    poids = markowitz_portfolio(choix)

    # Résultat formaté
    resultat = []
    for i, row in choix.iterrows():
        resultat.append({
            "opcvm": row["libelle"],
            "rendement": round(row["Rendement_annuel_simple"] * 100, 2),
            "poids": round(poids[list(choix.index).index(i)] * 100, 2)
        })

    # 🔹 Ici au lieu de jsonify, on affiche directement la page aleatoire.html
    return render_template(
        "aleatoire.html",
        profil=profil,
        portefeuille=resultat
    )













@app.route('/opcvm')
def page_opcvm():
    table = get_table_name()
    conn = get_db_connection()
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()

    total_opcvm = len(df)
    moyenne_perf = round(df['performance_totale'].mean(), 2)
    total_clusters = df['cluster'].nunique()
    vol_moyenne = round(df['volatilite_annualisee'].mean(), 2)
    opcvm_list = df['denomination_opcvm'].unique()

    return render_template('index1.html',
                           total_opcvm=total_opcvm,
                           moyenne_perf=moyenne_perf,
                           total_clusters=total_clusters,
                           vol_moyenne=vol_moyenne,
                           opcvm_list=opcvm_list)
    
chromedriver_path = r"C:\Users\HP\Downloads\chromedriver-win64 (1)\chromedriver-win64\chromedriver.exe"
def get_indices():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-dev-shm-usage")

    service = ChromeService(executable_path=chromedriver_path)
    driver = webdriver.Chrome(service=service, options=options)

    indices_list = []

    try:
        driver.get("https://www.casablanca-bourse.com/fr/live-market/marche-cash/indices")
        time.sleep(5)  # attendre le chargement complet

        tickers_elements = driver.find_elements(By.XPATH, "//p[contains(@class,'uppercase') and contains(@class,'text-sm')]")
        values_elements = driver.find_elements(By.XPATH, "//div[contains(@class,'inline-flex') and contains(@class,'items-center')]")

        for i in range(len(tickers_elements)):
            ticker = tickers_elements[i].text.strip()
            price = values_elements[i].find_element(By.XPATH, ".//p/span[@dir='ltr']").text.strip()
            change = values_elements[i].find_elements(By.XPATH, ".//div/p/span[@dir='ltr']")[0].text.strip()

            # Récupérer l’icône (SVG use href)
            try:
                icon_use = values_elements[i].find_element(By.XPATH, ".//svg/use").get_attribute("href")
                # Exemple : "/icons.svg#bourse-up" ou "/icons.svg#bourse-down"
                icon = icon_use if icon_use else ""
            except:
                icon = ""

            indices_list.append({
                "ticker": ticker,
                "price": price,
                "change": change,
                "icon": icon
            })

    finally:
        driver.quit()

    return indices_list


@app.route("/")
def house():
    indices = get_indices()
    return render_template("OPCVM.html", indices=indices)


@app.route('/jrb', methods=['GET', 'POST'])
def home1():
    if request.method == 'POST':
        profil = request.form.get('profil')
        panier = request.form.getlist('panier')

        if not panier:
            return render_template(
                'jrb.html',
                opcvm_list=get_liste_opcvm(),
                erreur="Veuillez sélectionner au moins un OPCVM.",
                profil_sel=profil,
                panier=[],
                poids_opti=None
            )

        # Chargement des données Excel
        df_rendements = pd.read_excel(EXCEL_PATH, sheet_name='Rendements Alignes', index_col=0)
        df_covariance = pd.read_excel(EXCEL_PATH, sheet_name='Matrice Covariance', index_col=0)

        # Vérification que tous les OPCVM du panier sont présents
        for opc in panier:
            if opc not in df_rendements.columns or opc not in df_covariance.index:
                return render_template(
                    'jrb.html',
                    opcvm_list=get_liste_opcvm(),
                    erreur=f"L'OPCVM '{opc}' n'est pas disponible dans les données.",
                    profil_sel=profil,
                    panier=panier,
                    poids_opti=None
                )

        # Filtrer uniquement les OPCVM sélectionnés
        df_rendements_selection = df_rendements[panier]
        df_covariance_selection = df_covariance.loc[panier, panier]

        try:
            poids_opt = optimiser_portefeuille(df_rendements_selection, profil)
        except Exception as e:
            return render_template(
                'jrb.html',
                opcvm_list=get_liste_opcvm(),
                erreur=f"Erreur de calcul des pondérations : {str(e)}",
                profil_sel=profil,
                panier=panier,
                poids_opti=None
            )

        return render_template(
            'jrb.html',
            profil_sel=profil,
            panier=panier,
            poids_opti=poids_opt.round(4).to_dict(),
            erreur=None
        )

    return render_template('jrb.html', opcvm_list=get_liste_opcvm(),
                           erreur=None, profil_sel=None, panier=[], poids_opti=None)

def get_liste_opcvm():
    if not os.path.exists(EXCEL_PATH):
        raise FileNotFoundError(f"Fichier Excel introuvable : {EXCEL_PATH}")
    df_rendements = pd.read_excel(EXCEL_PATH, sheet_name='Rendements Alignes', index_col=0)
    return list(df_rendements.columns)

@app.route('/ACCEUIL')
def accueil():
    return render_template('OPCVM.html')

def calcul_ponderations_markowitz(returns, profil, taux_sans_risque=0.0214):
    import numpy as np
    from scipy.optimize import minimize

    mu = returns.mean() * 52
    sigma = returns.cov() * 52
    n = len(mu)

    def rendement(w): 
        return np.dot(w, mu)

    def risque(w): 
        return np.sqrt(np.dot(w.T, np.dot(sigma, w)))

    def sharpe(w): 
        return -(rendement(w) - taux_sans_risque) / risque(w)

    contraintes = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bornes = tuple((1e-6, 1) for _ in range(n))  # min > 0
    w0 = np.array([1/n] * n)

    if profil.lower() == 'dynamique':
        fonction_obj = sharpe
    elif profil.lower() == 'modere':
        rendement_max = np.max(mu)
        contraintes.append({'type': 'ineq', 'fun': lambda w: np.dot(w, mu) - rendement_max * 0.75})
        fonction_obj = risque
    elif profil.lower() == 'prudent':
        rendement_max = np.max(mu)
        contraintes.append({'type': 'ineq', 'fun': lambda w: np.dot(w, mu) - rendement_max * 0.5})
        fonction_obj = risque
    else:
        raise ValueError("Profil inconnu")

    result = minimize(fonction_obj, w0, method='SLSQP', bounds=bornes, constraints=contraintes)
    if not result.success:
        raise Exception(f"Optimisation échouée: {result.message}")

    return dict(zip(returns.columns, result.x))

#  Nouvelle fonction pour connecter Markowitz à /jrb
def optimiser_portefeuille(df_rendements_selection, profil):
    return pd.Series(
        calcul_ponderations_markowitz(df_rendements_selection, profil),
        index=df_rendements_selection.columns
    )

@app.route("/calcul_ponderations", methods=["POST"])
def calculer():
    try:
        data = request.json
        profil = data.get("profil")
        opcvm_list = data.get("opcvm")

        if not profil or not opcvm_list:
            return jsonify({"erreur": "Profil ou liste OPCVM manquants"}), 400

        returns = pd.read_excel(EXCEL_PATH, sheet_name="Rendements Alignes", index_col=0)
        manquants = [o for o in opcvm_list if o not in returns.columns]
        if manquants:
            return jsonify({"erreur": f"OPCVM absents du fichier: {manquants}"}), 400

        returns_filtre = returns[opcvm_list]
        ponderations = calcul_ponderations_markowitz(returns_filtre, profil)

        return jsonify(ponderations)

    except Exception as e:
        return jsonify({"erreur": str(e)}), 500
    

@app.route("/portefeuille_aleatoire", methods=["GET"])
def portefeuille_aleatoire():
    profil = request.args.get("profil", "").lower()
    nombreOpcvm = request.args.get("nombreOpcvm", "10").lower()

    if not os.path.exists(EXCEL_PATH):
        return jsonify({"erreur": "Fichier Excel introuvable."}), 500

    df_rendements = pd.read_excel(EXCEL_PATH, sheet_name='Rendements Alignes', index_col=0)
    opcvm_list = list(df_rendements.columns)

    if len(opcvm_list) == 0:
        return jsonify({"erreur": "Aucun OPCVM disponible."}), 400

    if nombreOpcvm == "all":
        opcvm_selectionnes = opcvm_list
    else:
        try:
            n = int(nombreOpcvm)
        except ValueError:
            return jsonify({"erreur": "Nombre d'OPCVM invalide."}), 400

        if n < 1 or n > len(opcvm_list):
            return jsonify({"erreur": f"Le nombre d'OPCVM doit être entre 1 et {len(opcvm_list)}."}), 400

        if profil == "prudent":
            n = min(n, 10)
        elif profil == "modere":
            n = min(n, 20)
        elif profil == "dynamique":
            pass
        else:
            return jsonify({"erreur": "Profil inconnu."}), 400

        opcvm_selectionnes = random.sample(opcvm_list, n)

    nb_final = len(opcvm_selectionnes)
    poids = np.random.dirichlet(np.ones(nb_final), size=1)[0]
    ponderations = {k: round(v, 4) for k, v in zip(opcvm_selectionnes, poids)}

    # Calcul rendement moyen pondéré
    rendements_sel = df_rendements[opcvm_selectionnes]
    rendements_moyens = rendements_sel.mean()  # moyenne par colonne OPCVM
    rendement_portefeuille = np.dot(poids, rendements_moyens)

    # Calcul volatilité (risque) portefeuille
    cov_mat = rendements_sel.cov()
    volatilite_portefeuille = np.sqrt(np.dot(poids.T, np.dot(cov_mat, poids)))

    return jsonify({
        "profil": profil,
        "nombre_opcvm": nb_final,
        "portefeuille": ponderations,
        "rendement": round(rendement_portefeuille, 6),  # rendement moyen
        "risque": round(volatilite_portefeuille, 6)    # volatilité
    })
def compute_indicators(df):
    """Ajoute MA50, MA200, EMA12, EMA26, RSI(14) à df (DataFrame prix ajustés)."""
    out = df.copy()
    out['MA50'] = out['Close'].rolling(window=50).mean()
    out['MA200'] = out['Close'].rolling(window=200).mean()
    out['EMA12'] = out['Close'].ewm(span=12, adjust=False).mean()
    out['EMA26'] = out['Close'].ewm(span=26, adjust=False).mean()
    # RSI
    delta = out['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    rs = ma_up / (ma_down + 1e-9)
    out['RSI14'] = 100 - (100 / (1 + rs))
    return out

def annualized_returns(returns):
    return returns.mean() * 252

def annualized_cov(returns):
    return returns.cov() * 252

def markowitz_weights(returns_df, profile='modere', rf=0.0214):
    """Renvoie pondérations Markowitz optimisées (long only, sum=1)."""
    mu = annualized_returns(returns_df)
    Sigma = annualized_cov(returns_df)
    n = len(mu)
    w0 = np.array([1/n]*n)
    bounds = tuple((1e-6,1) for _ in range(n))
    cons = ({'type':'eq', 'fun': lambda w: np.sum(w)-1})

    def port_return(w): return np.dot(w, mu)
    def port_risk(w): return np.sqrt(np.dot(w.T, np.dot(Sigma, w)))
    def neg_sharpe(w): return -(port_return(w)-rf)/port_risk(w)

    if profile.lower() == 'dynamique':
        obj = neg_sharpe
    else:
        # pour modere/prudent minimiser volatilité sous contrainte rendement minimal
        target = 0.75 * np.max(mu) if profile.lower()=='modere' else 0.5 * np.max(mu)
        cons = list(cons)
        cons.append({'type':'ineq', 'fun': lambda w: np.dot(w, mu) - target})
        obj = port_risk

    res = minimize(obj, w0, method='SLSQP', bounds=bounds, constraints=cons)
    if not res.success:
        raise Exception("Optimisation échouée: " + res.message)
    return dict(zip(returns_df.columns, res.x))

DATABASE_ACTIONS = "base_actions4.db"

def get_db_connection_actions():
    if not os.path.exists(DATABASE_ACTIONS):
        raise FileNotFoundError(f"La base de données SQLite n'existe pas : {DATABASE_ACTIONS}")
    conn = sqlite3.connect(DATABASE_ACTIONS)
    conn.row_factory = sqlite3.Row
    return conn

def get_table_name_actions():
    conn = get_db_connection_actions()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()
    if not tables:
        raise Exception("Aucune table trouvée dans la base de données.")
    return tables[0][0]

@app.route("/rechercheaction")
def rechercheaction():
    return render_template("rechercheaction.html")

@app.route("/resultataction")
def resultataction():
    nom = request.args.get("nom", "").strip()
    table = get_table_name_actions()
    conn = get_db_connection_actions()
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    conn.close()

    if nom not in df['libelle'].values:
        return render_template("resultataction.html", nom=nom, resultat=[], message="Aucune action trouvée.")

    cluster = df.loc[df['libelle']==nom, 'Cluster'].values[0]
    similaires = df[df['Cluster']==cluster]

    return render_template("resultataction.html", nom=nom, resultat=similaires.to_dict(orient='records'), message=None)


@app.route("/indice")
def indice():
    driver_path = r"C:\Users\HP\Downloads\chromedriver-win64 (1)\chromedriver-win64\chromedriver.exe"
    url = "https://www.casablanca-bourse.com/fr/live-market/marche-cash/indices"

    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")

    driver = webdriver.Chrome(service=Service(driver_path), options=options)
    driver.get(url)
    time.sleep(3)

    data = []
    rows = driver.find_elements(By.CSS_SELECTOR, "table tbody tr")

    for row in rows:
        cols = row.find_elements(By.TAG_NAME, "td")
        if len(cols) == 7:
            data.append([col.text.strip() for col in cols])

    driver.quit()

    df = pd.DataFrame(data, columns=["Indice", "Valeur", "Veille","Variation %", "Variation 31/12", "Plus haut", "Plus bas"])
    
    # convertir en dictionnaire pour l'envoyer au template
    indices_data = df.to_dict(orient="records")

    return render_template("indice.html", indices=indices_data)
app.secret_key = "secret_key"  # nécessaire pour flash messages

DOWNLOAD_FOLDER = "C:/Users/HP/Downloads/ASFIM_files"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
CHROMEDRIVER_PATH = r"C:\Users\HP\Downloads\chromedriver-win64 (1)\chromedriver-win64\chromedriver.exe"

def download_all_files():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    service = ChromeService(executable_path=CHROMEDRIVER_PATH)
    driver = webdriver.Chrome(service=service, options=options)

    url = "https://asfim.ma/publications/tableaux-des-performances/"
    driver.get(url)
    time.sleep(3)

    all_links = []

    while True:
        links = driver.find_elements(By.TAG_NAME, "a")
        for link in links:
            href = link.get_attribute("href")
            if href and href.lower().endswith(".xlsx") and href not in all_links:
                all_links.append(href)

        try:
            next_button = driver.find_element(By.ID, "DataTables_Table_0_next")
            if "disabled" in next_button.get_attribute("class"):
                break
            else:
                next_button.click()
                time.sleep(2)
        except:
            break

    driver.quit()

    downloaded_files = []
    for file_url in all_links:
        filename = os.path.join(DOWNLOAD_FOLDER, file_url.split("/")[-1])
        if not os.path.exists(filename):
            response = requests.get(file_url)
            with open(filename, "wb") as f:
                f.write(response.content)
        downloaded_files.append(filename)

    return downloaded_files

def send_email(to_email, file_path):
    EMAIL_ADDRESS = "ton_email@gmail.com"
    EMAIL_PASSWORD = "ton_mot_de_passe"  # ou utiliser un mot de passe d'application

    msg = EmailMessage()
    msg["Subject"] = "Dernier tableau ASFIM"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = to_email
    msg.set_content("Voici le dernier tableau des performances ASFIM.")

    with open(file_path, "rb") as f:
        file_data = f.read()
        file_name = os.path.basename(file_path)
    msg.add_attachment(file_data, maintype="application", subtype="vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename=file_name)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        smtp.send_message(msg)

@app.route("/À propos", methods=["GET", "POST"])
def propos():
    if request.method == "POST":
        email = request.form.get("email")
        if email:
            # Télécharger tous les fichiers et récupérer le dernier
            files = download_all_files()
            last_file = files[-1]  # dernier fichier téléchargé
            send_email(email, last_file)
            flash(f"Le dernier tableau a été envoyé à {email} !", "success")
        else:
            flash("Veuillez entrer un email valide.", "error")
        return redirect("/propos")
    return render_template("propos.html")
app.secret_key = "ton_secret_key"  # change ça pour plus de sécurité

# --- Création de la base users ---
def init_db():
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# --- Route portail utilisateur ---
@app.route("/portail", methods=["GET", "POST"])
def portail():
    if request.method == "POST":
        action = request.form.get("action")

        if action == "register":
            username = request.form["username"]
            email = request.form["email"]
            password = generate_password_hash(request.form["password"])

            try:
                conn = sqlite3.connect("users.db")
                c = conn.cursor()
                c.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)", 
                          (username, email, password))
                conn.commit()
                conn.close()
                flash(" Compte créé avec succès, connecte-toi maintenant.", "success")
            except sqlite3.IntegrityError:
                flash(" Nom d’utilisateur ou email déjà utilisé.", "danger")

        elif action == "login":
            email = request.form["email"]
            password = request.form["password"]

            conn = sqlite3.connect("users.db")
            c = conn.cursor()
            c.execute("SELECT id, username, password FROM users WHERE email = ?", (email,))
            user = c.fetchone()
            conn.close()

            if user and check_password_hash(user[2], password):
                session["user_id"] = user[0]
                session["username"] = user[1]
                flash(f" Bienvenue {user[1]} !", "success")
                return redirect(url_for("dashboard"))
            else:
                flash(" Email ou mot de passe incorrect.", "danger")

    return render_template("portail.html")

# --- Tableau de bord après connexion ---

@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect(url_for("portail"))
    return render_template("dashboard.html", username=session["username"])

# --- Déconnexion ---
@app.route("/logout")
def logout():
    session.clear()
    flash(" Déconnexion réussie.", "info")
    return redirect(url_for("portail"))
# Chemin du dossier où les fichiers ASFIM sont téléchargés
# Chemin du dossier ASFIM
DOWNLOAD_FOLDER = r"C:/Users/HP/Downloads/ASFIM_files"

def get_performance_info():
    files = [f for f in os.listdir(DOWNLOAD_FOLDER) if f.lower().endswith('.xlsx')]
    if not files:
        return {
            "count": 0,
            "last_file": None,
            "last_date": None,
            "table": pd.DataFrame()
        }
    files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(DOWNLOAD_FOLDER, x)), reverse=True)
    last_file = files[0]
    last_date = datetime.fromtimestamp(os.path.getmtime(os.path.join(DOWNLOAD_FOLDER, last_file))).strftime("%d/%m/%Y %H:%M")
    table = pd.read_excel(os.path.join(DOWNLOAD_FOLDER, last_file))
    return {
        "count": len(files),
        "last_file": last_file,
        "last_date": last_date,
        "table": table
    }

@app.route("/suivi")
def suivi():
    if "user_id" not in session:
        return redirect(url_for("portail"))

    perf_info = get_performance_info()
    # Passer le DataFrame au template
    return render_template(
        "suivi.html",
        perf_info=perf_info,
        username=session.get("username", "Utilisateur"),
        gender=session.get("gender", "homme")
    )
ACTIONS = [
    "ADH MC Equity", "ADI MC Equity", "AFI MC Equity", "AFM MC Equity", "AGMA MC Equity",
    "AKT MC Equity", "ALM MC Equity", "AOULA MC Equity", "ARD MC Equity", "ATH MC Equity",
    "ATL MC Equity", "ATW MC Equity", "BAL MC Equity", "BCI MC Equity", "BCP MC Equity",
    "BOA MC Equity", "CDM MC Equity", "CFG MC Equity", "CIH MC Equity", "CMA MC Equity",
    "CMG MC Equity", "CMT MC Equity", "COL MC Equity", "CRS MC Equity", "CSR MC Equity",
    "CTM MC Equity", "DHO MC Equity", "DRI MC Equity", "DWY MC Equity", "DYT MC Equity",
    "EQD MC Equity", "FBR MC Equity", "GAZ MC Equity", "HPS MC Equity", "IAM MC Equity",
    "IBC MC Equity", "IMO MC Equity", "INV MC Equity", "JET MC Equity", "LBV MC Equity",
    "LES MC Equity", "LHM MC Equity", "M2M MC Equity", "MAB MC Equity", "MDP MC Equity",
    "MIC MC Equity", "MLE MC Equity", "MNG MC Equity", "MOX MC Equity", "MSA MC Equity",
    "MUT MC Equity", "NEJ MC Equity", "NKL MC Equity", "PRO MC Equity", "RDS MC Equity",
    "REB MC Equity", "RIS MC Equity", "S2M MC Equity", "SAH MC Equity", "SBM MC Equity",
    "SID MC Equity", "SLF MC Equity", "SMI MC Equity", "SNA MC Equity", "SNP MC Equity",
    "SOT MC Equity", "SRM MC Equity", "STR MC Equity", "TGC MC Equity", "TMA MC Equity",
    "TQM MC Equity", "UMR MC Equity", "WAA MC Equity", "ZDJ MC Equity"
]
# Dossier contenant les fichiers des rendements
DOSSIER_ACTIONS = r"C:\Users\HP\Documents\actions_separees2"

# Fonction pour charger les rendements d'une action

def charger_rendements(action):
    fichier = os.path.join(DOSSIER_ACTIONS, f"{action}.xlsx")
    df = pd.read_excel(fichier)  
    return df['Rendement_log'].dropna()

# Fonction pour calculer les pondérations Markowitz
def calcul_ponderations_markowitz_stocks(returns, profil, taux_sans_risque=0.0214):
    mu = returns.mean() * 252
    sigma = returns.cov() * 252
    n = len(mu)

    def rendement(w): return np.dot(w, mu)
    def risque(w): return np.sqrt(np.dot(w.T, np.dot(sigma, w)))
    def sharpe(w): return -(rendement(w) - taux_sans_risque) / risque(w)

    contraintes = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bornes = tuple((1e-6, 1) for _ in range(n))
    w0 = np.array([1/n] * n)

    if profil.lower() == 'dynamique':
        fonction_obj = sharpe
    elif profil.lower() == 'modere':
        rendement_max = np.max(mu)
        contraintes.append({'type': 'ineq', 'fun': lambda w: np.dot(w, mu) - rendement_max * 0.75})
        fonction_obj = risque
    elif profil.lower() == 'prudent':
        rendement_max = np.max(mu)
        contraintes.append({'type': 'ineq', 'fun': lambda w: np.dot(w, mu) - rendement_max * 0.5})
        fonction_obj = risque
    else:
        raise ValueError("Profil inconnu")

    result = minimize(fonction_obj, w0, method='SLSQP', bounds=bornes, constraints=contraintes)
    if not result.success:
        raise Exception(f"Optimisation échouée: {result.message}")

    return dict(zip(returns.columns, result.x))

# Route portefeuille
@app.route("/portefeuil", methods=["GET", "POST"])
def portefeuil():
    ponderations = None
    if request.method == "POST":
        profil = request.form.get("profil")
        actions_selectionnees = request.form.getlist("actions")
        if actions_selectionnees:
            data = {a: charger_rendements(a) for a in actions_selectionnees}
            df_rendements = pd.DataFrame(data).dropna()
            ponderations = calcul_ponderations_markowitz_stocks(df_rendements, profil)
        return render_template(
            "portefeuil.html",
            actions=ACTIONS,
            selected_profil=profil,
            selected_actions=actions_selectionnees,
            ponderations=ponderations
        )
    return render_template("portefeuil.html", actions=ACTIONS, selected_actions=[], ponderations=None)

# Route résultat stocks (avec pondérations)
@app.route("/resultatstocks", methods=["POST"])
def resultatstocks():
    profil = request.form.get("profil")
    actions_selectionnees = request.form.getlist("actions")
    ponderations = None
    if actions_selectionnees:
        data = {a: charger_rendements(a) for a in actions_selectionnees}
        df_rendements = pd.DataFrame(data).dropna()
        ponderations = calcul_ponderations_markowitz_stocks(df_rendements, profil)
    return render_template(
        "resultatstocks.html",
        profil=profil,
        actions_selectionnees=actions_selectionnees,
        ponderations=ponderations
    )
DATA_FOLDER = r"C:\Users\HP\Documents\actions_separees2"  # dossier contenant les fichiers Excel

def load_data():
    """Charge les rendements moyens, volatilités et prix du 13/08/2025 depuis les fichiers Excel."""
    assets = {}
    target_date = pd.to_datetime("2025-08-13")

    for file in os.listdir(DATA_FOLDER):
        if file.endswith(".xlsx"):
            df = pd.read_excel(os.path.join(DATA_FOLDER, file))
            df.columns = df.columns.str.strip()

            # Trouver la colonne de dates
            date_col = [c for c in df.columns if "date" in c.lower()]
            if not date_col:
                continue
            date_col = date_col[0]

            # Assurer que la colonne est datetime
            if not np.issubdtype(df[date_col].dtype, np.datetime64):
                df[date_col] = pd.to_datetime(df[date_col])

            # Colonne _PX_LAST
            col_px = [c for c in df.columns if c.endswith("_PX_LAST")]
            if not col_px:
                continue
            col_px = col_px[0]

            name = col_px.replace("_PX_LAST", "")

            # Convertir en décimal
            mean_return = df["Rendement_annuel_simple"].iloc[-1] / 100
            volatility = df["Volatilite_annuelle"].iloc[-1] / 100

            # Extraire le prix pour target_date
            price_row = df[df[date_col] == target_date]
            price = price_row[col_px].values[0] if not price_row.empty else np.nan

            assets[name] = {
                "mean_return": mean_return,
                "volatility": volatility,
                "price": price
            }

    return assets

def markowitz_optimization(returns, cov_matrix, target_return):
    """Optimisation de Markowitz sous contrainte de rendement minimal (inégalité)."""
    n = len(returns)

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},             # somme des poids = 1
        {"type": "ineq", "fun": lambda w: np.dot(w, returns) - target_return}  # rendement >= cible
    )
    bounds = tuple((0, 1) for _ in range(n))
    init_guess = np.ones(n) / n

    result = minimize(portfolio_volatility, init_guess, bounds=bounds, constraints=constraints)
    if result.success:
        return result.x
    else:
        return None



EXCEL_FILE = r"C:\Users\HP\Documents\resultats_markowitz.xlsx"

# Lecture des données
mu = pd.read_excel(EXCEL_FILE, sheet_name="Rendements_annuels", index_col=0)["Rendement Annuel"]
Sigma = pd.read_excel(EXCEL_FILE, sheet_name="Covariance_annuelle", index_col=0)
actions = mu.index.tolist()

@app.route("/personalised", methods=["GET", "POST"])
def personalised():
    result = None
    error_msg = None
    sigma_real = None
    actual_return = None
    approx = False

    if request.method == "POST":
        selected_actions = request.form.getlist("actions")
        if not selected_actions:
            error_msg = "Veuillez sélectionner au moins une action."
        else:
            try:
                R_target = float(request.form.get("rendement"))
                sigma_max = float(request.form.get("volatilite"))
                min_weight = 0.01  # poids minimum pour chaque action
            except:
                error_msg = "Veuillez entrer des valeurs numériques correctes."
                return render_template("index.html", actions=actions, result=result, error_msg=error_msg)

            mu_sel = mu[selected_actions]
            Sigma_sel = Sigma.loc[selected_actions, selected_actions] + np.eye(len(selected_actions)) * 1e-8

            n = len(selected_actions)
            w = cp.Variable(n)

            # Objectif : minimiser la variance
            portfolio_variance = cp.quad_form(w, Sigma_sel.values)
            objective = cp.Minimize(portfolio_variance)

            # Contraintes
            constraints = [
                cp.sum(w) == 1,
                w >= min_weight,
                w @ mu_sel.values >= R_target,
                cp.quad_form(w, Sigma_sel.values) <= sigma_max**2
            ]

            prob = cp.Problem(objective, constraints)

            try:
                prob.solve()  # solveur par défaut (ECOS/OSQP)
            except:
                pass

            # Si aucune solution → approximation
            if w.value is None:
                approx = True
                w = cp.Variable(n)
                objective = cp.Maximize(w @ mu_sel.values)
                constraints = [
                    cp.sum(w) == 1,
                    w >= min_weight,
                    cp.quad_form(w, Sigma_sel.values) <= (1.2 * sigma_max)**2  # tolérance 20%
                ]
                prob = cp.Problem(objective, constraints)
                prob.solve()

            if w.value is None:
                error_msg = "Impossible de construire un portefeuille avec ces actions."
            else:
                weights = pd.Series(np.round(w.value, 6), index=selected_actions)
                actual_return = float(weights.values @ mu_sel.values)
                sigma_real = float(np.sqrt(weights.values @ Sigma_sel.values @ weights.values))

                result = weights.sort_values(ascending=False).to_frame("Pondération")
                result["Rendement attendu"] = actual_return
                result["Volatilité"] = sigma_real

                if approx:
                    error_msg = f"Portefeuille approximatif (Rendement réel : {actual_return:.2%}, Volatilité : {sigma_real:.2%})"

    return render_template("portefeuil.html", actions=actions, result=result,
                           error_msg=error_msg, sigma_real=sigma_real, actual_return=actual_return)


chromedriver_path = r"C:\Users\HP\Downloads\chromedriver-win64 (1)\chromedriver-win64\chromedriver.exe"
download_folder = "C:/Users/HP/Downloads/ASFIM_files"
os.makedirs(download_folder, exist_ok=True)


# --------------------------
# Fonctions utilitaires
# --------------------------
def extract_date_from_filename(filename):
    match = re.search(r"(20\d{2}[01]\d[0-3]\d)", filename)
    if match:
        return datetime.strptime(match.group(1), "%Y%m%d")
    match = re.search(r"(\d{2})-(\d{2})-(20\d{2})", filename)
    if match:
        return datetime.strptime("-".join(match.groups()), "%d-%m-%Y")
    return None


def get_latest_asfim_file():
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")

    service = ChromeService(executable_path=chromedriver_path)
    driver = webdriver.Chrome(service=service, options=options)

    url = "https://asfim.ma/publications/tableaux-des-performances/"
    driver.get(url)
    time.sleep(3)

    all_links = []
    while True:
        links = driver.find_elements("tag name", "a")
        for link in links:
            href = link.get_attribute("href")
            if href and href.lower().endswith(".xlsx") and href not in all_links:
                all_links.append(href)
        try:
            next_button = driver.find_element("id", "DataTables_Table_0_next")
            if "disabled" in next_button.get_attribute("class"):
                break
            else:
                next_button.click()
                time.sleep(2)
        except:
            break

    driver.quit()

    # Extraire la date
    files_with_dates = []
    for file_url in all_links:
        fname = file_url.split("/")[-1]
        fdate = extract_date_from_filename(fname)
        if fdate:
            files_with_dates.append((file_url, fdate))
    if not files_with_dates:
        if all_links:
            latest_file = all_links[-1]
        else:
            raise Exception("Aucun fichier disponible")
    else:
        latest_file = max(files_with_dates, key=lambda x: x[1])[0]

    filename = os.path.join(download_folder, latest_file.split("/")[-1])
    response = requests.get(latest_file)
    with open(filename, "wb") as f:
        f.write(response.content)

    return filename


# --------------------------
# Routes Flask
# --------------------------
@app.route("/performance")
def performance():
    try:
        file_path = get_latest_asfim_file()
        df = pd.read_excel(file_path, header=1)  # 2ème ligne comme header
        table_html = df.to_html(classes="table table-striped table-bordered", index=False)

        # Extraire la date depuis le nom du fichier
        fname = os.path.basename(file_path)
        fdate = extract_date_from_filename(fname)
        table_date = fdate.strftime("%d/%m/%Y") if fdate else "Date inconnue"

        flash("Fichier téléchargé avec succès", "success")
        return render_template("performance.html", table_html=table_html, table_date=table_date)
    except Exception as e:
        flash(f"Erreur : {e}", "danger")
        return render_template("performance.html", table_html=None, table_date=None)

@app.route("/download_performance")
def download_performance():
    try:
        file_path = get_latest_asfim_file()
        return send_file(file_path, as_attachment=True)
    except Exception as e:
        flash(f"Erreur téléchargement : {e}", "danger")
        return redirect(url_for('performance'))

def get_all_user_emails():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT email FROM users")
    emails = [row[0] for row in cursor.fetchall()]
    conn.close()
    return emails


def send_email_with_file(to_email, subject, body, file_path):
    from_email = "ton_email@gmail.com"
    password = "mot_de_passe_application"  # mot de passe d’application Gmail

    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    with open(file_path, "rb") as f:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(f.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition", f"attachment; filename={file_path.split('/')[-1]}")

    msg.attach(part)

    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(from_email, password)
    server.send_message(msg)
    server.quit()


def job_send_asfim_file():
    try:
        file_path = get_latest_asfim_file()
        emails = get_all_user_emails()
        for email in emails:
            send_email_with_file(
                to_email=email,
                subject="Nouveau fichier ASFIM",
                body="Bonjour, voici le dernier tableau des performances ASFIM.",
                file_path=file_path
            )
        print(" Emails envoyés avec succès")
    except Exception as e:
        print(" Erreur envoi email :", e)


@app.route("/notification")
def notification():
    return render_template("notification.html")


@app.route("/send_now")
def send_now():
    try:
        job_send_asfim_file()
        flash("Emails envoyés avec succès !", "success")
    except Exception as e:
        flash(f"Erreur envoi manuel : {e}", "danger")
    return redirect(url_for("notification"))


# --------------------------
# Scheduler
# --------------------------
scheduler = BackgroundScheduler()
scheduler.add_job(job_send_asfim_file, 'cron', hour=8, minute=30)  # tous les jours à 08h30
scheduler.start()
atexit.register(lambda: scheduler.shutdown())

if __name__ == "__main__":
    # Lancer l'application Flask normalement
    app.run(debug=True)