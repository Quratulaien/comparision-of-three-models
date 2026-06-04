"""
ui/app.py — Interactive Tkinter GUI for ML Classifier Comparison
Run: python ui/app.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# ── Train models ──────────────────────────────
iris   = load_iris()
X, y   = iris.data, iris.target
cn     = iris.target_names
fn     = iris.feature_names

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
sc = StandardScaler().fit(X_tr)
Xs_tr = sc.transform(X_tr)
Xs_te = sc.transform(X_te)

knn = KNeighborsClassifier(n_neighbors=5).fit(Xs_tr, y_tr)
dt  = DecisionTreeClassifier(max_depth=5, random_state=42).fit(Xs_tr, y_tr)
nb  = GaussianNB().fit(Xs_tr, y_tr)

MODELS = {"KNN (k=5)": knn, "Decision Tree": dt, "Naïve Bayes": nb}
ACCS   = {n: f"{accuracy_score(y_te, m.predict(Xs_te)):.2%}" for n,m in MODELS.items()}
COLORS_CLASS = ["#1a9e75", "#c87a1e", "#c0392b"]
EMOJI        = ["🌸", "🌿", "🌺"]

# ── Window ────────────────────────────────────
root = tk.Tk()
root.title("Iris Flower Classifier — KNN | Decision Tree | Naïve Bayes")
root.geometry("680x620")
root.configure(bg="#f8f7f5")
root.resizable(False, False)

FONT_H  = ("Segoe UI", 13, "bold")
FONT_B  = ("Segoe UI", 11, "bold")
FONT_N  = ("Segoe UI", 11)
FONT_S  = ("Segoe UI", 9)
BG      = "#f8f7f5"
CARD_BG = "#ffffff"
ACCENT  = "#3266ad"

# ── Header ────────────────────────────────────
header = tk.Frame(root, bg=ACCENT, pady=14)
header.pack(fill="x")
tk.Label(header, text="🌸 Iris Flower Classifier", font=("Segoe UI", 16, "bold"),
         bg=ACCENT, fg="white").pack()
tk.Label(header, text="KNN  ·  Decision Tree  ·  Naïve Bayes  |  Iris Dataset",
         font=FONT_S, bg=ACCENT, fg="#c8d9f0").pack()

# ── Accuracy badges ──────────────────────────
badge_frame = tk.Frame(root, bg=BG, pady=8)
badge_frame.pack(fill="x", padx=20)
for i, (name, acc) in enumerate(ACCS.items()):
    f = tk.Frame(badge_frame, bg=CARD_BG, relief="flat", bd=0,
                 highlightthickness=1, highlightbackground="#e0ddd8")
    f.grid(row=0, column=i, padx=6, pady=4, sticky="ew", ipadx=10, ipady=6)
    badge_frame.columnconfigure(i, weight=1)
    tk.Label(f, text=name, font=("Segoe UI", 9, "bold"), bg=CARD_BG, fg="#555").pack()
    tk.Label(f, text=acc,  font=("Segoe UI", 15, "bold"), bg=CARD_BG, fg=ACCENT).pack()
    tk.Label(f, text="test accuracy", font=FONT_S, bg=CARD_BG, fg="#999").pack()

# ── Input card ──────────────────────────────
input_card = tk.LabelFrame(root, text="  Enter Measurements  ", font=FONT_B,
                            bg=CARD_BG, fg="#333", relief="flat", bd=0,
                            highlightthickness=1, highlightbackground="#e0ddd8",
                            padx=18, pady=14)
input_card.pack(fill="x", padx=20, pady=(4, 0))

sliders = {}
slider_vals = {}
labels_info = [
    ("Sepal Length", "cm", 4.3, 7.9, 5.8),
    ("Sepal Width",  "cm", 2.0, 4.4, 3.1),
    ("Petal Length", "cm", 1.0, 6.9, 3.8),
    ("Petal Width",  "cm", 0.1, 2.5, 1.2),
]

for i, (lbl, unit, mn, mx, default) in enumerate(labels_info):
    row = i // 2
    col = (i % 2) * 4
    tk.Label(input_card, text=f"{lbl} ({unit})", font=FONT_N,
             bg=CARD_BG, fg="#444", anchor="w").grid(row=row*2, column=col, columnspan=2,
             sticky="w", padx=(0 if col==0 else 20, 0))
    var = tk.DoubleVar(value=default)
    slider_vals[lbl] = var
    val_lbl = tk.Label(input_card, text=f"{default:.1f}", font=FONT_B,
                       bg=CARD_BG, fg=ACCENT, width=5)
    val_lbl.grid(row=row*2, column=col+2, columnspan=2, sticky="e",
                 padx=(0 if col==0 else 20, 0))

    def _make_cb(v, l):
        def cb(val): l.config(text=f"{float(val):.1f}")
        return cb

    s = ttk.Scale(input_card, from_=mn, to=mx, variable=var,
                  orient="horizontal", length=250,
                  command=_make_cb(var, val_lbl))
    s.grid(row=row*2+1, column=col, columnspan=4, sticky="ew",
           pady=(2, 10), padx=(0 if col==0 else 20, 0))
    sliders[lbl] = s
    input_card.columnconfigure(col, weight=1)

# Preset buttons
preset_frame = tk.Frame(input_card, bg=CARD_BG)
preset_frame.grid(row=4, column=0, columnspan=8, pady=(0, 4))
tk.Label(preset_frame, text="Quick presets:", font=FONT_S, bg=CARD_BG, fg="#888").pack(side="left", padx=(0,8))

PRESETS = [
    ("Setosa sample",     5.1, 3.5, 1.4, 0.2),
    ("Versicolor sample", 6.0, 2.9, 4.5, 1.5),
    ("Virginica sample",  6.7, 3.1, 5.6, 2.4),
]

def load_preset(sl, sw, pl, pw):
    slider_vals["Sepal Length"].set(sl)
    slider_vals["Sepal Width"].set(sw)
    slider_vals["Petal Length"].set(pl)
    slider_vals["Petal Width"].set(pw)

for pname, sl, sw, pl, pw in PRESETS:
    tk.Button(preset_frame, text=pname, font=FONT_S, bg="#eef3fb", fg=ACCENT,
              relief="flat", bd=0, cursor="hand2", padx=8, pady=3,
              command=lambda a=sl,b=sw,c=pl,d=pw: load_preset(a,b,c,d)).pack(side="left", padx=3)

# Algorithm selector + Classify button
ctrl_frame = tk.Frame(root, bg=BG, pady=6)
ctrl_frame.pack(fill="x", padx=20)
tk.Label(ctrl_frame, text="Algorithm:", font=FONT_B, bg=BG, fg="#444").pack(side="left")
algo_var = tk.StringVar(value="KNN (k=5)")
for name in MODELS:
    ttk.Radiobutton(ctrl_frame, text=name, variable=algo_var, value=name).pack(side="left", padx=8)

tk.Button(ctrl_frame, text="  Classify  ▶", font=("Segoe UI", 11, "bold"),
          bg=ACCENT, fg="white", relief="flat", bd=0, cursor="hand2",
          padx=16, pady=6, activebackground="#185FA5", activeforeground="white",
          command=lambda: classify()).pack(side="right")

# ── Result card ───────────────────────────────
result_card = tk.LabelFrame(root, text="  Prediction Result  ", font=FONT_B,
                             bg=CARD_BG, fg="#333", relief="flat", bd=0,
                             highlightthickness=1, highlightbackground="#e0ddd8",
                             padx=18, pady=14)
result_card.pack(fill="x", padx=20, pady=(4, 4))

species_var = tk.StringVar(value="—")
conf_var    = tk.StringVar(value="Enter measurements and click Classify")

species_lbl = tk.Label(result_card, textvariable=species_var,
                        font=("Segoe UI", 20, "bold"), bg=CARD_BG, fg="#333")
species_lbl.pack()
tk.Label(result_card, textvariable=conf_var,
         font=FONT_S, bg=CARD_BG, fg="#777").pack(pady=(2, 8))

prob_frame = tk.Frame(result_card, bg=CARD_BG)
prob_frame.pack(fill="x")
prob_bars = []
for i, cname in enumerate(cn):
    row = tk.Frame(prob_frame, bg=CARD_BG)
    row.pack(fill="x", pady=2)
    tk.Label(row, text=f"{EMOJI[i]} {cname.capitalize()}", font=FONT_S,
             bg=CARD_BG, fg="#555", width=14, anchor="w").pack(side="left")
    canvas = tk.Canvas(row, height=14, bg="#eeece8", bd=0,
                       highlightthickness=0, width=340)
    canvas.pack(side="left", padx=6)
    pct_lbl = tk.Label(row, text="0%", font=FONT_S, bg=CARD_BG, fg="#777", width=5)
    pct_lbl.pack(side="left")
    prob_bars.append((canvas, pct_lbl))

# All-models row
all_frame = tk.Frame(result_card, bg=CARD_BG)
all_frame.pack(fill="x", pady=(8, 0))
tk.Label(all_frame, text="All models:", font=FONT_S, bg=CARD_BG, fg="#888").pack(side="left", padx=(0,8))
all_labels = {}
for name in MODELS:
    f = tk.Frame(all_frame, bg="#eef3fb", padx=8, pady=3)
    f.pack(side="left", padx=3)
    tk.Label(f, text=name, font=("Segoe UI", 8), bg="#eef3fb", fg="#888").pack()
    lbl = tk.Label(f, text="—", font=("Segoe UI", 9, "bold"), bg="#eef3fb", fg=ACCENT)
    lbl.pack()
    all_labels[name] = lbl

# ── Classify function ────────────────────────
def classify():
    vals = np.array([
        slider_vals["Sepal Length"].get(),
        slider_vals["Sepal Width"].get(),
        slider_vals["Petal Length"].get(),
        slider_vals["Petal Width"].get(),
    ]).reshape(1, -1)
    sample = sc.transform(vals)
    algo = algo_var.get()
    model = MODELS[algo]
    pred = model.predict(sample)[0]
    probs = model.predict_proba(sample)[0]

    species_var.set(f"{EMOJI[pred]}  {cn[pred].capitalize()}")
    species_lbl.config(fg=COLORS_CLASS[pred])
    conf_var.set(f"Confidence: {probs[pred]:.1%}  |  Algorithm: {algo}")

    for i, (canvas, pct_lbl) in enumerate(prob_bars):
        p = probs[i]
        canvas.delete("all")
        w = int(340 * p)
        if w > 0:
            canvas.create_rectangle(0, 0, w, 14, fill=COLORS_CLASS[i], outline="")
        pct_lbl.config(text=f"{p:.0%}")

    for name, mdl in MODELS.items():
        p2 = mdl.predict(sample)[0]
        all_labels[name].config(text=cn[p2].capitalize(), fg=COLORS_CLASS[p2])

# ── Status bar ───────────────────────────────
tk.Label(root, text="Iris Dataset · 150 samples · 4 features · 3 classes  |  Train 80%  Test 20%  |  StandardScaler preprocessing",
         font=FONT_S, bg="#e8e6e0", fg="#888", anchor="w", padx=10, pady=4
         ).pack(fill="x", side="bottom")

root.mainloop()
