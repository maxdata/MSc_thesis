#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd

import os
import flask
import json
import unidecode

from flask import Flask, session, redirect, url_for, render_template, request

# # Load data
df = pd.read_csv("data/filtered_df.csv")
isco_codes = pd.read_csv("data/isco translations it sp uk nl fr de - Sheet1.csv")[["ISCO_08_Code",
                                                                                   "NL ISCO "]].set_index(
    "ISCO_08_Code").to_dict()["NL ISCO "]

with open("data/index_to_isco.json") as f:
    isco = json.load(f)
    isco = {int(k): int(v) for k, v in isco.items()}

index_to_name = {k: isco_codes[v] for k, v in isco.items()}

df_grouped = df.groupby("candidate_id")

# CNN-LSTM, LSTM, CNN,
examples = {"Gezondheidszorg" : [(7112518, 235), (8399558, 138), (8526077, 234)],
            "Financiën" : [(5270318, 143), (7293359, 4), (7378736, 72)],
            "Klantenservice" : [(8093077, 186), (8443697, 192), (6911669, 187)]}

example_to_name = {0: "CNN-LSTM", 1: "LSTM", 2: "CNN"}


# # Flask

# Connect to the database and setup the app
app = Flask(__name__)
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'

@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        session["email"] = request.form["email"]
        session["user_type"] = request.form['submit_button']
        session["example"] = 0
        return redirect(url_for("sliders"))
    else:
        session["example"] = 0
        return render_template("/index.html")

@app.route('/sliders', methods=["GET", "POST"])
def sliders():

    if request.method == "POST":
        session["example"] += 1

        top_left = int(request.form["topleft"].replace("option", ""))
        top_right = int(request.form["topright"].replace("option", ""))
        bottom = int(request.form["bottom"].replace("option", ""))
        general = int(request.form["general"].replace("option", ""))

        file_name = f"results/ratings/{session['email']}_{example_to_name[session['example'] - 1]}_{session['user_type']}.txt"

        with open(file_name, "w+") as f:
            f.write(f"User: {session['email']}\nSubject: {session['user_type']}\nModel: {example_to_name[session['example'] - 1]}\n\n" +
                    f"- Feature attention: {top_left}\n- Temporal attention: {top_right}\n- Spatiotemporal attention: {bottom}\n" +
                    f"- General: {general}")

        print("Stored ratings:", top_left, top_right, bottom, general)

        # os.system(f"scp {file_name} ec2-user@172.17.0.2:/home/ec2-user/randstad/results")

    if session["example"] >= len(examples[session["user_type"]]):
        return render_template("/done.html")

    current_example = examples[session["user_type"]][session["example"]]
    session["pred"] = index_to_name[current_example[1]]

    df = df_grouped.get_group(current_example[0]).reset_index().drop(["index",
                                                                      "candidate_id"], axis=1).iloc[:-1]

    static_features = pd.DataFrame(df[["location", "skills", "certificates", "licenses", "languages"]].iloc[0].T)
    df = df.drop(["location", "skills", "certificates", "licenses", "languages"], axis=1)

    df = df[["time_spent", "isco_functie_niveau", "education", "company_name", "function_id", "isco_code4", "text"]]

    df.index.name = "Baan nummer"
    df.index += 1
    session["max_job"] = len(df.index)
    df = df.T
    df.index = ["Dagen gewerkt", "Werkniveau", "Opleidingsniveau", "Bedrijf", "Functie", "Isco code", "CV"]
    df = df.loc[["Isco code", "Functie", "Bedrijf", "Opleidingsniveau", "Dagen gewerkt", "Werkniveau", "CV"]]

    static_features.index = ["Postcode", "Vaardigheden", "Certificaten", "Rijbewijzen", "Talen"]
    static_feautres = static_features.loc[["Vaardigheden", "Certificaten", "Talen", "Rijbewijzen", "Postcode"]]

    session["df"] = df.to_html(classes="table", escape=False)
    session["static_features"] = static_features.to_html(header=False,
                                                         classes=["table", "static_data"])

    return render_template("/sliders.html",
                           n=session["example"] + 1,
                           dataframe=session["df"],
                           static_features=session["static_features"],
                           pred=session["pred"],
                           max_job=session["max_job"])

@app.route("/show_results", methods=["GET", "POST"])
def show_results():
    image = f"../static/{unidecode.unidecode(session['user_type'])}/{example_to_name[session['example']]}.png"

    return render_template("/show_results.html",
                           image=image,
                           n=session["example"] + 1,
                           dataframe=session["df"],
                           static_features=session["static_features"],
                           pred=session["pred"],
                           max_job=session["max_job"])

@app.route("/store_results", methods=["POST"])
def store_results():

    r = request.get_json()

    values = dict(zip(["isco code", "function id", "company", "education", "days worked",
                       "isco level", "CV", "skills", "certificates", "languages", "licenses",
                       "location"], r["values"]))

    file_name = f"results/sliders/{session['email']}_{example_to_name[session['example']]}_{session['user_type']}.txt"

    with open(file_name, "w+") as f:
        f.write(f"User: {session['email']}\nSubject: {session['user_type']}\nModel: {example_to_name[session['example']]}\n\n" +
                f"Values: {values}")

    print("Stored sliders:", values)

    # os.system(f"scp {file_name} ec2-user@172.17.0.2:/home/ec2-user/randstad/results")

    return redirect(url_for("show_results"))

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
