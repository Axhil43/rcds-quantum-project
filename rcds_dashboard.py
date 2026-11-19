"""Dark research dashboard joining RCDS, MetaMap, ML, Arduino and CUDA-Q layers.
Run with: streamlit run rcds_dashboard.py
All biological and risk values are synthetic research-demo outputs, not clinical predictions.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import serial
from serial.tools import list_ports
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
from rcds_metamap_engine import SensorReading, pathway_engine
from cudaq_rcds_sim import (backend_status, benchmark_ghz_comparison, ghz_chain,
                            ghz_statevector_probabilities, metabolic_classifier,
                            quantum_adduct_susceptibility, run_vqe_curve, select_target)
import cudaq

st.set_page_config(page_title="RCDS × MetaMap Research Lab", page_icon="🧪", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
:root { color-scheme: dark; }
.stApp { background: radial-gradient(ellipse at 14% 0%, #112b42 0%, #091521 43%, #060b12 100%); color:#e8f1f8; }
.block-container { max-width:1540px; padding-top:2rem; padding-bottom:3rem; }
h1,h2,h3 { font-family:'Space Grotesk',sans-serif !important; letter-spacing:-.025em; }
p, label, [data-testid="stMarkdownContainer"] { font-family:'DM Sans',sans-serif; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#0b1725,#08111b); border-right:1px solid #233d53; }
[data-testid="stSidebar"] [data-testid="stExpander"] { background:#0d1d2b; border:1px solid #233d53; border-radius:13px; margin-bottom:.55rem; }
[data-testid="stMetric"] { min-height:132px; height:auto; padding:20px 21px; border-radius:18px; background:linear-gradient(145deg,rgba(24,48,69,.88),rgba(10,23,35,.88)); border:1px solid rgba(103,173,214,.27); box-shadow:0 10px 34px rgba(0,0,0,.22), inset 0 1px 0 rgba(255,255,255,.035); overflow:visible; }
[data-testid="stMetricLabel"] { white-space:normal !important; overflow:visible !important; text-overflow:clip !important; color:#a9c0d2 !important; line-height:1.35 !important; font-size:.91rem !important; }
[data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"] { overflow:visible !important; width:100% !important; }
[data-testid="stMetricLabel"] [data-testid="stMarkdownContainer"] p { white-space:normal !important; overflow:visible !important; text-overflow:unset !important; display:block !important; line-height:1.35 !important; }
[data-testid="stMetricValue"] { color:#f4f8fc !important; font-family:'Space Grotesk',sans-serif; font-size:clamp(1.55rem,2.15vw,2.25rem) !important; margin-top:.45rem; }
.kpi-card { min-height:134px; height:auto; padding:19px 21px; border-radius:18px; background:linear-gradient(145deg,rgba(24,48,69,.9),rgba(10,23,35,.92)); border:1px solid rgba(103,173,214,.33); box-shadow:0 10px 34px rgba(0,0,0,.2),inset 0 1px 0 rgba(255,255,255,.045),0 0 22px rgba(49,155,198,.045); }
.kpi-label { display:block; color:#a9c0d2; font:500 .9rem 'DM Sans',sans-serif; line-height:1.4; white-space:normal; overflow-wrap:anywhere; min-height:2.55rem; }
.kpi-value { color:#f4f8fc; font:600 clamp(1.55rem,2.1vw,2.15rem) 'Space Grotesk',sans-serif; line-height:1.25; margin-top:.4rem; }
[data-testid="stMetricDelta"] { white-space:normal; }
.stTabs [data-baseweb="tab-list"] { gap:8px; border-bottom:1px solid #21394e; }
.stTabs [data-baseweb="tab"] { background:rgba(15,31,46,.75); border:1px solid transparent; border-radius:12px 12px 0 0; padding:12px 17px; color:#adc0cf; }
.stTabs [aria-selected="true"] { background:#142a3d !important; border-color:#2c526e !important; color:#f0f8ff !important; }
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] { border:1px solid rgba(91,139,172,.2); border-radius:16px; background:rgba(10,23,35,.55); padding:5px; }
.stButton button, [data-testid="stDownloadButton"] button { border-radius:11px; border:1px solid #3c789c; background:linear-gradient(135deg,#173d57,#14516a); color:#f4fbff; font-weight:600; min-height:2.7rem; }
.stButton button:hover, [data-testid="stDownloadButton"] button:hover { border-color:#67c8eb; color:white; box-shadow:0 0 20px rgba(68,182,226,.18); }
[data-testid="stSlider"] [role="slider"] { accent-color:#58c6df; }
.risk-badge { display:inline-flex; align-items:center; gap:.55rem; padding:.52rem .92rem; margin:.2rem 0 1.15rem; border-radius:999px; border:1px solid; font:600 .88rem 'DM Sans',sans-serif; letter-spacing:.01em; }
.risk-dot { width:.55rem; height:.55rem; border-radius:50%; display:inline-block; }
.risk-low { color:#9de6c2; background:rgba(40,157,112,.13); border-color:rgba(82,217,157,.42); }
.risk-low .risk-dot { background:#55dda0; box-shadow:0 0 11px #55dda0; }
.risk-moderate { color:#f4d58c; background:rgba(181,129,36,.13); border-color:rgba(233,190,92,.43); }
.risk-moderate .risk-dot { background:#f0bf56; box-shadow:0 0 11px #f0bf56; }
.risk-high { color:#ffb17f; background:rgba(192,87,44,.14); border-color:rgba(246,134,88,.45); }
.risk-high .risk-dot { background:#ff8a54; box-shadow:0 0 11px #ff8a54; }
.risk-critical { color:#ff9aab; background:rgba(186,56,78,.16); border-color:rgba(245,93,119,.52); }
.risk-critical .risk-dot { background:#ff5d7c; box-shadow:0 0 11px #ff5d7c; }
.status-pill { display:inline-flex; align-items:center; gap:.4rem; padding:.48rem .82rem; margin:.15rem 0 .65rem; border:1px solid; border-radius:999px; background:rgba(10,24,36,.8); font:600 .82rem 'DM Sans',sans-serif; box-shadow:0 0 18px rgba(74,194,212,.08); }
.hero-panel { position:relative; overflow:hidden; padding:1.35rem 1.5rem; margin:.4rem 0 1.2rem; border:1px solid rgba(92,181,211,.25); border-radius:22px; background:linear-gradient(112deg,rgba(17,44,64,.92),rgba(10,23,36,.92) 55%,rgba(9,34,37,.9)); box-shadow:0 16px 48px rgba(0,0,0,.2),inset 0 1px 0 rgba(255,255,255,.04); }
.hero-panel:after { content:''; position:absolute; width:260px; height:260px; right:-100px; top:-170px; border-radius:50%; background:rgba(48,199,185,.13); filter:blur(8px); }
.hero-eyebrow { color:#68d5d2; text-transform:uppercase; letter-spacing:.16em; font:700 .68rem 'DM Sans',sans-serif; }
.hero-title { margin:.42rem 0 .32rem; color:#f1f7fb; font:600 clamp(1.4rem,2.3vw,2.1rem) 'Space Grotesk',sans-serif; letter-spacing:-.04em; }
.hero-copy { max-width:900px; color:#9fb7c8; font:400 .96rem/1.65 'DM Sans',sans-serif; }
.flow-track { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:1.15rem 0 1.25rem; }
.flow-stage { position:relative; min-height:112px; padding:15px 16px; border:1px solid rgba(112,166,194,.22); border-radius:16px; background:linear-gradient(145deg,rgba(18,39,56,.92),rgba(10,22,33,.86)); }
.flow-stage:not(:last-child):after { content:'›'; position:absolute; z-index:2; right:-12px; top:36px; width:20px; height:28px; color:#59c8cd; font-size:1.7rem; font-weight:600; text-align:center; }
.flow-index { color:#55d4c2; font:700 .7rem 'DM Sans',sans-serif; letter-spacing:.12em; }
.flow-name { margin:.35rem 0 .25rem; color:#e8f2f8; font:600 .98rem 'Space Grotesk',sans-serif; }
.flow-detail { color:#91a9ba; font:400 .78rem/1.5 'DM Sans',sans-serif; }
.explain-panel { padding:1.1rem 1.2rem; border-left:2px solid #47c8c3; border-radius:0 14px 14px 0; background:linear-gradient(90deg,rgba(30,91,96,.16),rgba(15,31,44,.5)); color:#b4c6d2; line-height:1.65; }
.lab-step { padding:.85rem 1rem; border:1px solid rgba(92,181,211,.2); border-radius:14px; background:rgba(15,33,48,.72); margin:.3rem 0 .8rem; }
.lab-step b { color:#6bd4d2; letter-spacing:.04em; }
.mini-note { padding:.75rem .9rem; border:1px solid rgba(113,158,183,.18); border-radius:12px; background:rgba(11,24,35,.62); color:#9eb4c3; font-size:.84rem; line-height:1.55; }
@media(max-width:900px) { .flow-track { grid-template-columns:repeat(2,minmax(0,1fr)); } .flow-stage:nth-child(2):after { display:none; } }
@media(max-width:560px) { .flow-track { grid-template-columns:1fr; } .flow-stage:after { display:none !important; } }
.section-kicker { text-transform:uppercase; letter-spacing:.12em; color:#7fb7d5; font-size:.72rem; font-weight:700; }
.stCaption, [data-testid="stCaptionContainer"] { color:#91aabc; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Training 1,500 synthetic pathway states…")
def train_predictors():
    rng = np.random.default_rng(2026)
    enzymes = ["GALT", "ALDOB", "GALK1", "UGP2", "PDH"]
    rows, enzyme_labels, risk_labels = [], [], []
    for _ in range(1500):
        if rng.random() < .72:
            deficient = str(rng.choice(enzymes))
            eff = {name: float(rng.uniform(.65, 1.0)) for name in enzymes}
            eff[deficient] = float(rng.uniform(.02, .12))
        else:
            deficient = "Healthy / mixed"
            eff = {name: float(rng.uniform(.68, 1.0)) for name in enzymes}
        concentration = float(rng.uniform(0, 220))
        glucose, fructose, galactose = (float(rng.uniform(.1, hi)) for hi in (10, 8, 6))
        result = pathway_engine(eff["GALT"], concentration, float(rng.uniform(6.8, 8)),
                                float(rng.uniform(20, 42)), glucose, fructose, galactose, eff)
        rows.append({
            "Carcinogen (uM)": concentration, "Glucose inflow": glucose,
            "Fructose inflow": fructose, "Galactose inflow": galactose,
            **{name + " efficiency": eff[name] for name in enzymes},
            "F1P buildup": result["F1P buildup"], "Gal1P buildup": result["Gal1P buildup"],
            "Galactitol load": result["Galactitol osmotic load"], "ATP depletion": result["ATP depletion"],
            "Lipogenic load": result["Lipogenic load"], "UDP-glucuronate": result["UDP-glucuronate pool"],
            "Sigma-sum": result["Sigma-sum stress score"],
            "DNA adduct index": result["Uncleared DNA adduct risk (scaled)"],
        })
        enzyme_labels.append(deficient)
        if concentration < 20 and result["Sigma-sum stress score"] < 2.5:
            risk_labels.append("Low")
        elif concentration < 85 and result["Uncleared DNA adduct risk (scaled)"] < 8:
            risk_labels.append("Moderate")
        else:
            risk_labels.append("High")
    X = pd.DataFrame(rows)
    Xt, Xv, yet, yev, yrt, yrv = train_test_split(
        X, enzyme_labels, risk_labels, test_size=.25, random_state=42, stratify=enzyme_labels)
    enzyme_model = RandomForestClassifier(n_estimators=160, min_samples_leaf=2,
                                          class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    risk_model = RandomForestClassifier(n_estimators=160, min_samples_leaf=2,
                                        class_weight="balanced_subsample", random_state=43, n_jobs=-1)
    enzyme_model.fit(Xt, yet)
    risk_model.fit(Xt, yrt)
    return {"columns": X.columns.tolist(), "enzyme": enzyme_model, "risk": risk_model,
            "enzyme_accuracy": accuracy_score(yev, enzyme_model.predict(Xv)),
            "risk_accuracy": accuracy_score(yrv, risk_model.predict(Xv)), "n": len(X)}


def read_arduino(port):
    with serial.Serial(port, baudrate=9600, timeout=4) as connection:
        line = connection.readline().decode("ascii").strip()
    values = [float(part) for part in line.split(",")]
    if len(values) != 8:
        raise ValueError("Expected R,G,B,blank_R,blank_G,blank_B,pH,temp_C")
    return SensorReading(tuple(map(int, values[:3])), tuple(map(int, values[3:6])),
                         values[6], values[7])


def pathway_table(result):
    return pd.DataFrame([
        ("Glycolysis", result["HK1 flux"], result["PFK1 flux"]),
        ("Fructolysis", result["KHK flux"], result["ALDOB-TKFC flux"]),
        ("Leloir pathway", result["GALK1 flux"], result["GALE flux"]),
        ("Polyol shunt", result["AKR1B1 flux"], result["SORD flux"]),
        ("Pyruvate oxidation", result["PDH flux"], result["PDH flux"]),
        ("TCA entry", result["TCA entry flux"], result["TCA entry flux"]),
        ("De novo lipogenesis", result["ACC flux"], result["FASN flux"]),
        ("Hepatic glucuronidation", result["UGP2 flux"], result["UGT1A1 clearance capacity"]),
    ], columns=["Pathway", "Entry / upstream", "Exit / downstream"])


def wiring_plot():
    positions = {"Uno": (.08,.52), "TCS34725": (.42,.82), "pH board": (.42,.52),
                 "DS18B20": (.42,.22), "4.7 kΩ pull-up": (.76,.10)}
    links = [("Uno","TCS34725","5V→VIN, GND→GND"),("Uno","TCS34725","A4 SDA, A5 SCL"),
             ("Uno","pH board","5V, GND, A0←AO"),("Uno","DS18B20","5V, GND, D2←DATA"),
             ("DS18B20","4.7 kΩ pull-up","DATA↔5V")]
    fig = go.Figure()
    for a,b,label in links:
        fig.add_trace(go.Scatter(x=[positions[a][0],positions[b][0]],y=[positions[a][1],positions[b][1]],
                                 mode="lines",line=dict(color="#5b82a8",width=2),
                                 hovertemplate=label+"<extra></extra>",showlegend=False))
    for label,(x,y) in positions.items():
        fig.add_trace(go.Scatter(x=[x],y=[y],mode="markers+text",text=[label],textposition="middle right",
                                 marker=dict(size=23,color="#4aa3df"),textfont=dict(color="white"),
                                 hoverinfo="text",showlegend=False))
    fig.update_layout(template="plotly_dark",height=340,margin=dict(l=5,r=20,t=5,b=5),
                      xaxis=dict(visible=False,range=[0,.98]),yaxis=dict(visible=False,range=[0,1]),
                      paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    return fig


def pathway_network_figure(result, concentration):
    positions = {
        "RCDS Griess sensor": (0.04, 0.55), "Carcinogen load": (0.21, 0.55),
        "Glycolysis": (0.20, 0.91), "Fructolysis": (0.20, 0.75),
        "Leloir pathway": (0.20, 0.35), "Gal1P pool": (0.42, 0.35),
        "UGP2": (0.61, 0.35), "UDP-glucuronate": (0.78, 0.35),
        "UGT1A1 clearance": (0.93, 0.55), "Polyol shunt": (0.43, 0.10),
        "PDH": (0.43, 0.91), "TCA": (0.65, 0.91),
        "De novo lipogenesis": (0.83, 0.80),
    }
    edges = [
        ("RCDS Griess sensor","Carcinogen load","assay signal"),
        ("Carcinogen load","UGT1A1 clearance","detoxification load"),
        ("Glycolysis","PDH","pyruvate"),
        ("Fructolysis","PDH","triose carbon"),
        ("PDH","TCA","acetyl-CoA"),
        ("TCA","De novo lipogenesis","citrate export"),
        ("Leloir pathway","Gal1P pool","Gal1P"),
        ("Gal1P pool","UGP2","competitive inhibition"),
        ("UGP2","UDP-glucuronate","UDP-glucose → UDPGA"),
        ("UDP-glucuronate","UGT1A1 clearance","co-substrate"),
        ("Leloir pathway","Polyol shunt","overflow"),
    ]
    fig = go.Figure()
    for source, target, label in edges:
        x0,y0 = positions[source]
        x1,y1 = positions[target]
        is_inhibition = label == "competitive inhibition"
        if is_inhibition:
            # Route this inhibitory edge above the flux line and mark its endpoint.
            yi = y0 + .13
            fig.add_trace(go.Scatter(x=[x0,x1], y=[yi,yi], mode="lines",
                line=dict(color="#ff637e", width=2.5, dash="dash"),
                hovertemplate="Gal1P competitively inhibits UGP2 · pool "
                    + f"{result['Gal1P buildup']:.3f}<extra></extra>", showlegend=False))
            fig.add_annotation(x=x1-.008,y=yi,ax=x0+.01,ay=yi,
                xref="x",yref="y",axref="x",ayref="y",showarrow=True,arrowhead=3,
                arrowsize=1.05,arrowwidth=1.1,arrowcolor="#ff637e",text="")
            fig.add_annotation(x=(x0+x1)/2,y=yi+.035,showarrow=False,
                text="Gal1P inhibits UGP2",font=dict(color="#ff8295",size=9),
                bgcolor="rgba(20,11,20,.78)",bordercolor="rgba(255,99,126,.35)",borderpad=2)
            continue
        fig.add_trace(go.Scatter(x=[x0,x1],y=[y0,y1],mode="lines",
            line=dict(color="rgba(105,157,190,.46)",width=1.7),
            hovertemplate=label+"<extra></extra>",showlegend=False))
    node_colors = []
    sizes = []
    for name in positions:
        if name == "Gal1P pool":
            node_colors.append("#fa6f82" if result["Gal1P buildup"] > .25 else "#d88d91")
            sizes.append(30 + min(18, result["Gal1P buildup"]*15))
        elif name in ("RCDS Griess sensor","Carcinogen load"):
            node_colors.append("#e4a74c" if concentration > 80 else "#4bbad2")
            sizes.append(29)
        elif name == "UGT1A1 clearance":
            node_colors.append("#55d6a1")
            sizes.append(34)
        elif name == "UGP2":
            node_colors.append("#eb7c83")
            sizes.append(31)
        else:
            node_colors.append("#518eb2")
            sizes.append(27)
    short_labels = {
        "RCDS Griess sensor":"RCDS", "Carcinogen load":"Carcinogen",
        "Glycolysis":"Glycolysis", "Fructolysis":"Fructolysis",
        "Leloir pathway":"Leloir", "Gal1P pool":"Gal1P",
        "UGP2":"UGP2", "UDP-glucuronate":"UDPGA",
        "UGT1A1 clearance":"UGT1A1", "Polyol shunt":"Polyol",
        "PDH":"PDH", "TCA":"TCA", "De novo lipogenesis":"DNL",
    }
    labels = [f"<b>{short_labels[name]}</b>" for name in positions]
    hover = []
    for name in positions:
        if name == "Gal1P pool":
            hover.append(f"Gal1P buildup: {result['Gal1P buildup']:.3f}")
        elif name == "UGP2":
            hover.append(f"UGP2 flux: {result['UGP2 flux']:.4f}")
        elif name == "UDP-glucuronate":
            hover.append(f"UDP-glucuronate pool: {result['UDP-glucuronate pool']:.4f}")
        elif name == "UGT1A1 clearance":
            hover.append(f"UGT1A1 capacity: {result['UGT1A1 clearance capacity']:.4f}")
        else:
            hover.append(name)
    fig.add_trace(go.Scatter(
        x=[p[0] for p in positions.values()],y=[p[1] for p in positions.values()],
        mode="markers+text",text=labels,textposition="bottom center",
        textfont=dict(color="#e6f1f8",size=10),marker=dict(size=sizes,color=node_colors,
            line=dict(color="rgba(222,245,255,.7)",width=1.2),opacity=.96),
        customdata=hover,hovertemplate="%{customdata}<extra></extra>",showlegend=False))
    fig.update_layout(template="plotly_dark",height=630,margin=dict(l=15,r=20,t=25,b=10),
        xaxis=dict(visible=False,range=(-.04,1.04)),yaxis=dict(visible=False,range=(0,1.06)),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Metabolic pathway map · hover for modeled pools and fluxes",
                   font=dict(size=15,color="#dceaf3")),
        annotations=fig.layout.annotations)
    return fig


def pathway_comparison_figure(table):
    display_names = {
        "Hepatic glucuronidation":"Glucuronidation",
        "De novo lipogenesis":"Lipogenesis",
        "Pyruvate oxidation":"Pyruvate oxidation",
        "TCA entry":"TCA entry",
        "Polyol shunt":"Polyol shunt",
        "Leloir pathway":"Leloir",
        "Fructolysis":"Fructolysis",
        "Glycolysis":"Glycolysis",
    }
    table = table.copy()
    table["Pathway"] = table["Pathway"].map(display_names).fillna(table["Pathway"])
    melted = table.melt(id_vars="Pathway",var_name="Step",value_name="Normalized flux")
    fig = px.bar(melted,x="Normalized flux",y="Pathway",color="Step",barmode="group",
        orientation="h",template="plotly_dark",
        color_discrete_map={"Entry / upstream":"#54c4d7","Exit / downstream":"#e5a958"},
        text_auto=".2f",title="Upstream vs downstream flux")
    fig.update_traces(marker_line_width=0,textposition="outside",cliponaxis=False,
        hovertemplate="%{y}<br>%{fullData.name}: %{x:.3f}<extra></extra>")
    fig.update_layout(height=630,margin=dict(l=8,r=48,t=55,b=20),
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h",y=1.08,x=0),xaxis_title="Normalized flux",
        yaxis=dict(categoryorder="array",categoryarray=list(reversed(table["Pathway"])),
                   gridcolor="rgba(140,170,190,.12)"))
    return fig


def chart_layout(fig, height=330):
    fig.update_layout(template="plotly_dark",height=height,
        paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=18,r=18,t=58,b=35),
        font=dict(family="DM Sans, sans-serif",color="#d9e7f0"),
        title_font=dict(size=15,color="#eaf4fa"))
    return fig


def main():
    st.title("🧪 RCDS × MetaMap Research Lab")
    st.caption("Optical/electrochemical sensing · metabolic coupling · quantum simulation · synthetic ML")
    with st.sidebar:
        st.header("Control room")
        with st.expander("Sensor Inputs", expanded=True):
            mode = st.radio("Input mode", ["Simulated RCDS Sensor", "Live Arduino Uno Serial Port"])
            basis = st.radio("Simulated input basis", ["Absorbance", "Nitrosamine concentration (uM)"],
                             disabled=mode.startswith("Live"))
            absorbance_slider = st.slider("Griess absorbance (AU)", 0., 1.2, .22, .01,
                                          disabled=mode.startswith("Live") or basis != "Absorbance")
            concentration_slider = st.slider("Nitrosamine concentration (uM)", 0., 250., 58., 1.,
                                             disabled=mode.startswith("Live") or basis != "Nitrosamine concentration (uM)")
            ph = st.slider("Assay pH", 5., 9., 7.4, .1)
            temp = st.slider("Assay temperature (°C)", 10., 50., 25., .5)
        with st.expander("Dietary Inflows", expanded=False):
            st.caption("Normalized model units")
            glucose = st.slider("Glucose", 0., 10., 5., .1)
            fructose = st.slider("Fructose", 0., 8., 2., .1)
            galactose = st.slider("Galactose", 0., 6., 1., .1)
        with st.expander("Enzyme Efficiencies", expanded=False):
            st.caption("Percent of reference activity")
            eff = {name: st.slider(name, 1, 100, default) / 100
                   for name, default in [("GALT",100),("ALDOB",100),("GALK1",100),("UGP2",100),("PDH",100)]}
        reading = None
        if mode.startswith("Live"):
            ports = [p.device for p in list_ports.comports()] or ["/dev/cu.usbmodemXXXX"]
            port = st.selectbox("Arduino serial port", ports)
            if st.button("Read one Arduino sample", type="primary"):
                try:
                    st.session_state["arduino_reading"] = read_arduino(port)
                    st.success("Serial sample received")
                except Exception as error:
                    st.error("Serial read failed: " + str(error))
            reading = st.session_state.get("arduino_reading")
            if reading is None:
                st.info("Connect the Uno and request a sample. Firmware sends one CSV line per second.")

    if mode.startswith("Live") and reading:
        absorbance, concentration = reading.estimate_concentration_uM()
        ph, temp = reading.pH, reading.temperature_C
    elif mode.startswith("Live"):
        absorbance, concentration = 0., 0.
    elif basis == "Absorbance":
        absorbance = absorbance_slider
        concentration = absorbance / 3.8e3 * 1e6
    else:
        concentration = concentration_slider
        absorbance = concentration * 3.8e3 / 1e6
    if not mode.startswith("Live"):
        red = int(np.clip(round(255 * 10 ** (-absorbance)), 1, 255))
        reading = SensorReading((red,red,red),(255,255,255),ph,temp)
    result = pathway_engine(eff["GALT"], concentration, ph, temp, glucose, fructose, galactose, eff)
    stress = float(np.clip(result["Sigma-sum stress score"] / 6.2 * 100, 0, 100))
    detox = float(np.clip(result["UGT1A1 clearance capacity"] / .5828 * 100, 0, 100))
    try:
        quantum_score = quantum_adduct_susceptibility(
            concentration, eff["GALT"], max(0,1-result["ATP depletion"]),
            min(1,result["UDP-glucuronate pool"]))
    except Exception as error:
        quantum_score = float(np.clip(result["Uncleared DNA adduct risk (scaled)"]/20,0,1))
        st.warning("CUDA-Q inference unavailable; scaled model fallback used: " + str(error))
    if quantum_score >= .80 or stress >= 80:
        risk_label, risk_class = "Critical Risk", "critical"
    elif quantum_score >= .60 or stress >= 60:
        risk_label, risk_class = "High Risk", "high"
    elif quantum_score >= .35 or stress >= 35:
        risk_label, risk_class = "Moderate Risk", "moderate"
    else:
        risk_label, risk_class = "Low Risk", "low"
    st.markdown(f'<div class="risk-badge risk-{risk_class}"><span class="risk-dot"></span>{risk_label}</div>',
                unsafe_allow_html=True)
    cards = st.columns(4, gap="medium")
    kpis = [
        ("Cellular Stress Index", f"{stress:.1f} / 100"),
        ("Carcinogen Load", f"{concentration:.1f} µM"),
        ("UDP-Glucuronate Detox Capacity", f"{detox:.1f}%"),
        ("Quantum Adduct Risk Score", f"{quantum_score:.3f}"),
    ]
    for card, (label, value) in zip(cards, kpis):
        with card:
            st.markdown(f'<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>',
                        unsafe_allow_html=True)
    st.caption(f"Assay A={absorbance:.3f} AU  ·  pH {ph:.2f}  ·  {temp:.1f} °C  ·  illustrative model units")
    st.markdown("""
    <section class="hero-panel">
      <div class="hero-eyebrow">From optical signal to computational readout</div>
      <div class="hero-title">One instrument. Three models. A traceable risk pathway.</div>
      <div class="hero-copy">The Arduino Uno acquires the assay and environmental signals. MetaMap translates the estimated carcinogen load and enzyme state into pathway pools. A four-qubit CUDA-Q feature circuit encodes four resulting values and returns the illustrative adduct-susceptibility score shown above.</div>
      <div class="flow-track">
        <div class="flow-stage"><div class="flow-index">01 · ACQUIRE</div><div class="flow-name">RCDS + Arduino Uno</div><div class="flow-detail">RGB / absorbance, pH and temperature → serial data</div></div>
        <div class="flow-stage"><div class="flow-index">02 · ESTIMATE</div><div class="flow-name">Sensor calibration</div><div class="flow-detail">Blank-corrected optical signal → estimated carcinogen µM</div></div>
        <div class="flow-stage"><div class="flow-index">03 · MODEL</div><div class="flow-name">MetaMap pathways</div><div class="flow-detail">Enzyme efficiencies → ATP, UDPGA and adduct-risk features</div></div>
        <div class="flow-stage"><div class="flow-index">04 · ENCODE</div><div class="flow-name">CUDA-Q feature circuit</div><div class="flow-detail">Carcinogen, GALT, ATP and UDPGA → quantum expectation score</div></div>
      </div>
    </section>
    """, unsafe_allow_html=True)
    t1,t2,t3,t4 = st.tabs(["RCDS + Metabolic Coupling","ML Enzyme & Risk Predictor",
                          "CUDA-Q Quantum Lab","Arduino Uno Hardware & Firmware"])

    with t1:
        st.markdown('<div class="section-kicker">Mechanistic pathway map</div>', unsafe_allow_html=True)
        st.subheader("Sensor-to-clearance coupling")
        st.markdown('<div class="explain-panel"><b>How the quantum score connects to RCDS:</b> the sensor estimates carcinogen concentration; the pathway model estimates detoxification context; the metabolic quantum circuit receives four normalized inputs—carcinogen load, GALT activity, ATP ratio and UDP-glucuronate. Its output is an illustrative circuit score. The GHZ and H₂ experiments in the Quantum Lab demonstrate separate quantum workflows; neither changes the assay reading.</div>', unsafe_allow_html=True)
        st.caption("Hover nodes and links to inspect modeled fluxes and pools. The red dashed link shows Gal1P inhibition of UGP2.")
        table = pathway_table(result)
        network_col, comparison_col = st.columns([1.42, 1], gap="large")
        with network_col:
            st.plotly_chart(pathway_network_figure(result, concentration), width="stretch",
                            config={"displayModeBar": False, "scrollZoom": True})
        with comparison_col:
            st.plotly_chart(pathway_comparison_figure(table), width="stretch",
                            config={"displayModeBar": False, "scrollZoom": True})
        no_inhibition = .9*eff["UGP2"]*result["GALE flux"]/(.35+result["GALE flux"]+1e-9)
        inhibition = 100*(1-result["UGP2 flux"]/max(no_inhibition,1e-9))
        status_col, metrics_col = st.columns([1,2])
        with status_col:
            st.markdown(f'<div class="risk-badge risk-{"high" if inhibition >= 35 else "moderate"}"><span class="risk-dot"></span>UGP2 suppression · {inhibition:.1f}%</div>', unsafe_allow_html=True)
        with metrics_col:
            st.caption(f"Gal1P pool {result['Gal1P buildup']:.3f}  →  UGP2 flux {result['UGP2 flux']:.4f}  →  UDP-glucuronate pool {result['UDP-glucuronate pool']:.4f}")
        with st.expander("Pathway flux table", expanded=False):
            st.dataframe(table.set_index("Pathway"), width="stretch")

    with t2:
        st.subheader("Synthetic enzyme and carcinogen-tier predictors")
        models = train_predictors()
        values = {"Carcinogen (uM)":concentration,"Glucose inflow":glucose,
                  "Fructose inflow":fructose,"Galactose inflow":galactose,
                  **{name+" efficiency":eff[name] for name in ["GALT","ALDOB","GALK1","UGP2","PDH"]},
                  "F1P buildup":result["F1P buildup"],"Gal1P buildup":result["Gal1P buildup"],
                  "Galactitol load":result["Galactitol osmotic load"],"ATP depletion":result["ATP depletion"],
                  "Lipogenic load":result["Lipogenic load"],"UDP-glucuronate":result["UDP-glucuronate pool"],
                  "Sigma-sum":result["Sigma-sum stress score"],
                  "DNA adduct index":result["Uncleared DNA adduct risk (scaled)"]}
        X = pd.DataFrame([[values[c] for c in models["columns"]]],columns=models["columns"])
        a,b,c = st.columns(3)
        a.metric("Training states","1,500")
        b.metric("Predicted deficient enzyme",str(models["enzyme"].predict(X)[0]))
        c.metric("Predicted carcinogen tier",str(models["risk"].predict(X)[0]))
        a,b = st.columns(2)
        a.metric("Enzyme validation accuracy",f"{models['enzyme_accuracy']:.1%}")
        b.metric("Risk-tier validation accuracy",f"{models['risk_accuracy']:.1%}")
        importance = pd.DataFrame({"Feature":models["columns"],
            "Enzyme model":models["enzyme"].feature_importances_,
            "Risk model":models["risk"].feature_importances_}).melt(
                id_vars="Feature",var_name="Model",value_name="Importance")
        fig = px.bar(importance,x="Importance",y="Feature",color="Model",barmode="group",
                     orientation="h",template="plotly_dark",title="RandomForest feature importances")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
                          yaxis={"categoryorder":"total ascending"},height=650)
        st.plotly_chart(fig,width="stretch")
        st.caption("Validation applies only to this synthetic dataset; this is not a clinically trained model.")

    with t3:
        st.markdown('<div class="section-kicker">Quantum simulation studio · three complementary experiments</div>', unsafe_allow_html=True)
        st.subheader("CUDA-Q research workflow")
        try:
            status = backend_status()
            label_color = "#45d2a0" if status["nvidia_available"] else "#67cde0"
            st.markdown(f'<div class="status-pill" style="border-color:{label_color};color:{label_color}">● {status["backend"]}  ·  {status["status"]}</div>', unsafe_allow_html=True)
            st.caption(f"CUDA-Q {getattr(cudaq,'__version__','unknown')} · active target `{status['target']}`")
        except Exception as error:
            st.warning("CUDA-Q target unavailable: " + str(error))
        st.markdown('<div class="explain-panel"><b>Why three quantum workflows?</b> GHZ validates statevector execution and scaling. H₂ VQE demonstrates a quantum-classical chemistry algorithm. The metabolic feature circuit is the RCDS-linked experiment: it encodes the current sensor-derived carcinogen load and MetaMap context into a four-qubit circuit whose expectation value contributes to the KPI above.</div>', unsafe_allow_html=True)
        st.markdown('<div class="flow-track"><div class="flow-stage"><div class="flow-index">01 · VALIDATE</div><div class="flow-name">GHZ statevector</div><div class="flow-detail">Entanglement, probabilities and backend scaling</div></div><div class="flow-stage"><div class="flow-index">02 · DEMONSTRATE</div><div class="flow-name">H₂ UCCSD VQE</div><div class="flow-detail">Optimizer + quantum expectation values across bond lengths</div></div><div class="flow-stage"><div class="flow-index">03 · CONNECT</div><div class="flow-name">RCDS feature circuit</div><div class="flow-detail">Live MetaMap features encoded as rotation angles</div></div><div class="flow-stage"><div class="flow-index">04 · INTERPRET</div><div class="flow-name">Score + limitations</div><div class="flow-detail">Observable-based prototype output, not validated prediction</div></div></div>', unsafe_allow_html=True)
        ghz_tab, h2_tab, link_tab = st.tabs(["01 · GHZ statevector", "02 · H₂ UCCSD VQE", "03 · RCDS quantum link"])

        with ghz_tab:
            st.markdown('<div class="lab-step"><b>STAGE 01 / STATEVECTOR VALIDATION</b><br>Prepare a GHZ state, inspect its ideal statevector distribution, then measure how the full statevector simulation scales with qubit count.</div>', unsafe_allow_html=True)
            control_a, control_b, control_c = st.columns([1,1,1.2], gap="medium")
            with control_a: n = st.slider("GHZ qubit count",2,14,6,key="ghz_qubits")
            with control_b: shots = st.slider("Measurement shots",128,4096,1024,128,key="ghz_shots")
            with control_c:
                st.write("")
                run_sample = st.button("▶  Run statevector + shot sample", type="primary", width="stretch")
            circuit_col, prob_col = st.columns([1,1], gap="large")
            with circuit_col:
                st.markdown("**Circuit · H followed by a nearest-neighbor CNOT chain**")
                try: st.code(cudaq.draw(ghz_chain, n), language="text", line_numbers=False)
                except Exception as error: st.warning("Circuit drawing unavailable: " + str(error))
            if run_sample:
                try:
                    counts = cudaq.sample(ghz_chain,n,shots_count=shots)
                    st.session_state["ghz_probabilities"] = {str(state):count/shots for state,count in counts.items()}
                    st.session_state["ghz_probability_qubits"] = n
                    st.session_state["ghz_statevector_probabilities"] = ghz_statevector_probabilities(n)
                except Exception as error: st.error("GHZ execution failed: " + str(error))
            probabilities = st.session_state.get("ghz_statevector_probabilities")
            with prob_col:
                if probabilities:
                    q = st.session_state.get("ghz_probability_qubits", n)
                    hist = go.Figure(go.Bar(x=list(probabilities),y=list(probabilities.values()),
                        marker=dict(color=["#46c5d7" if key == "0"*q else "#4ed5a3" for key in probabilities],line=dict(color="rgba(230,245,255,.22)",width=1)),
                        text=[f"{value:.3f}" for value in probabilities.values()],textposition="outside",
                        hovertemplate="State %{x}<br>Probability %{y:.4f}<extra></extra>"))
                    hist.update_layout(title=f"Ideal statevector · {q} qubits",xaxis_title="Computational basis state",yaxis_title="Probability",yaxis=dict(range=[0,1.12],gridcolor="rgba(150,180,200,.13)"))
                    st.plotly_chart(chart_layout(hist,350),width="stretch",config={"displayModeBar":False})
                    counts = st.session_state.get("ghz_probabilities", {})
                    st.caption("Statevector check: 50% |0…0⟩ + 50% |1…1⟩. Shot frequencies for the selected sample are " + (", ".join(f"{state}: {prob:.1%}" for state,prob in counts.items()) if counts else "not yet collected") + ".")
                else: st.info("Run the circuit to inspect its statevector probabilities and finite-shot sample.")
            st.markdown("**Backend scaling · 2 to 20 qubits**")
            if st.button("Run statevector scaling benchmark",key="benchmark_ghz"):
                try:
                    with st.spinner("Measuring qpp-cpu and checking cuStateVec availability…"):
                        st.session_state["ghz_timings"] = benchmark_ghz_comparison()
                except Exception as error: st.error("Benchmark failed: " + str(error))
            timings = st.session_state.get("ghz_timings")
            if timings:
                sizes,seconds = zip(*timings["cpu"])
                scaling = go.Figure(go.Scatter(x=sizes,y=seconds,mode="lines+markers",name="Measured qpp-cpu",line=dict(color="#56c7d9",width=3),marker=dict(size=7),hovertemplate="%{x} qubits<br>%{y:.5g} s<extra>CPU</extra>"))
                gpu_sizes,gpu_seconds = zip(*timings["gpu"])
                scaling.add_trace(go.Scatter(x=gpu_sizes,y=gpu_seconds,mode="lines+markers",name=timings["gpu_label"],line=dict(color="#4ed5a3",width=3,dash="dash" if timings["gpu_projected"] else "solid"),marker=dict(size=7),hovertemplate="%{x} qubits<br>%{y:.5g} s<extra>GPU curve</extra>"))
                scaling.update_layout(title="Statevector runtime · logarithmic scale",xaxis_title="Qubits",yaxis_title="Runtime (s)",yaxis=dict(type="log",gridcolor="rgba(150,180,200,.13)"),xaxis=dict(dtick=2,gridcolor="rgba(150,180,200,.08)"))
                st.plotly_chart(chart_layout(scaling,390),width="stretch",config={"displayModeBar":False})
                if timings["gpu_projected"]: st.markdown('<div class="mini-note">This host has no NVIDIA GPU target. The cyan curve is measured qpp-cpu; the dashed green cuStateVec curve is a labeled projection only.</div>',unsafe_allow_html=True)
            else: st.caption("Run the benchmark to populate the interactive log-scale comparison.")

        with h2_tab:
            st.markdown('<div class="lab-step"><b>STAGE 02 / QUANTUM-CHEMISTRY DEMONSTRATION</b><br>STO-3G molecular integrals are mapped to a 15-term Jordan–Wigner Hamiltonian. CUDA-Q evaluates a Hartree–Fock-initialized UCCSD ansatz while COBYLA optimizes its amplitudes at each bond length.</div>', unsafe_allow_html=True)
            h2a,h2b,h2c = st.columns(3)
            h2a.markdown('<div class="mini-note"><b>MODEL</b><br>H₂ · 2 electrons · 4 spin orbitals · STO-3G</div>',unsafe_allow_html=True)
            h2b.markdown('<div class="mini-note"><b>QUANTUM CIRCUIT</b><br>Hartree–Fock |1100⟩ + UCCSD singles and doubles</div>',unsafe_allow_html=True)
            h2c.markdown('<div class="mini-note"><b>CLASSICAL LOOP</b><br>COBYLA minimizes ⟨H(R)⟩; exact sector diagonalization is overlaid</div>',unsafe_allow_html=True)
            if st.button("Compute H₂ potential-energy surface",key="compute_vqe",type="primary"):
                try:
                    with st.spinner("Building STO-3G Hamiltonians and optimizing UCCSD over 0.3–2.5 Å…"):
                        st.session_state["vqe_curve"] = run_vqe_curve()
                except Exception as error: st.error("VQE calculation failed: " + str(error))
            curve = st.session_state.get("vqe_curve")
            if curve:
                distances,vqe_energies,exact_energies = curve
                pes = go.Figure()
                pes.add_trace(go.Scatter(x=distances,y=exact_energies,mode="lines",name="Exact diagonalization",line=dict(color="#9cabb9",dash="dash",width=2),hovertemplate="%{x:.2f} Å<br>%{y:.6f} Ha<extra>Exact</extra>"))
                pes.add_trace(go.Scatter(x=distances,y=vqe_energies,mode="lines+markers",name="CUDA-Q UCCSD VQE",line=dict(color="#42c8d2",width=3),marker=dict(size=5),hovertemplate="%{x:.2f} Å<br>%{y:.6f} Ha<extra>VQE</extra>"))
                pes.add_vline(x=.74,line_dash="dot",line_color="#4ed5a3",annotation_text="equilibrium ≈ 0.74 Å",annotation_position="top")
                pes.update_layout(title="H₂ potential-energy surface · STO-3G",xaxis_title="H–H bond distance (Å)",yaxis_title="Ground-state energy (Hartree)",legend=dict(orientation="h",y=1.12),xaxis=dict(gridcolor="rgba(150,180,200,.08)"),yaxis=dict(gridcolor="rgba(150,180,200,.13)"))
                st.plotly_chart(chart_layout(pes,460),width="stretch",config={"displayModeBar":False,"scrollZoom":True})
                st.caption(f"Computed {len(distances)} bond distances. Minimum on this sampled grid: {distances[int(np.argmin(exact_energies))]:.2f} Å; maximum VQE–exact deviation: {np.max(np.abs(vqe_energies-exact_energies)):.2e} Ha.")
            else: st.info("Compute the surface to compare optimized UCCSD energies with exact diagonalization.")

        with link_tab:
            st.markdown('<div class="lab-step"><b>STAGE 03 / RCDS-TO-CIRCUIT DATA PATH</b><br>This is the only quantum circuit in this lab that receives live values from the RCDS + MetaMap model. The Arduino streams sensor measurements; it does not run CUDA-Q. The host computer converts the measurements and model pools into rotation angles.</div>', unsafe_allow_html=True)
            feature_values = [float(np.clip(concentration/100,0,1)),float(np.clip(eff["GALT"],0,1)),float(np.clip(1-result["ATP depletion"],0,1)),float(np.clip(result["UDP-glucuronate pool"],0,1))]
            feature_names = ["Carcinogen load","GALT activity","ATP ratio","UDP-glucuronate"]
            feature_sources = [f"Sensor estimate · {concentration:.2f} µM / 100",f"Sidebar enzyme input · {eff['GALT']:.0%}",f"MetaMap · 1 − ATP depletion ({result['ATP depletion']:.3f})",f"MetaMap · UDPGA pool ({result['UDP-glucuronate pool']:.3f})"]
            feature_table = pd.DataFrame({"Circuit input":feature_names,"Source value":feature_sources,"Normalized feature":feature_values,"Ry angle (rad)":np.asarray(feature_values)*np.pi})
            left,right = st.columns([1.05,1],gap="large")
            with left:
                st.markdown("**Live feature encoding**")
                st.dataframe(feature_table,hide_index=True,width="stretch",column_config={"Normalized feature":st.column_config.ProgressColumn("Normalized feature",min_value=0,max_value=1,format="%.3f"),"Ry angle (rad)":st.column_config.NumberColumn("Ry angle (rad)",format="%.3f")})
                st.metric("Current quantum adduct susceptibility",f"{quantum_score:.3f}",help="Expectation-derived prototype score from the four encoded features; not a calibrated probability.")
            with right:
                st.markdown("**Live four-qubit feature circuit**")
                try: st.code(cudaq.draw(metabolic_classifier,*[v*np.pi for v in feature_values]),language="text",line_numbers=False)
                except Exception as error: st.warning("Circuit drawing unavailable: "+str(error))
            st.markdown('<div class="mini-note"><b>Interpretation:</b> this circuit provides a quantum feature transformation and expectation value for a research prototype. It does not measure carcinogens itself, and its score is not a validated biological probability. The sensor and mechanistic pathway model remain the source of the underlying RCDS and metabolic estimates.</div>',unsafe_allow_html=True)

    with t4:
        st.subheader("Arduino Uno wiring and firmware")
        st.markdown('<div class="explain-panel"><b>Why the Arduino Uno is in this system:</b> it is the low-cost sensor interface and data-acquisition controller. It powers/reads the color sensor, pH interface and temperature probe, then streams one CSV record over USB serial. The host computer performs Beer–Lambert conversion, pathway simulation, machine-learning inference and all CUDA-Q simulation. The Uno does not run the quantum circuit.</div>', unsafe_allow_html=True)
        st.markdown("Firmware streams `R,G,B,blank_R,blank_G,blank_B,pH,temp_C` at 9600 baud. The Python dashboard reads that record and passes the converted values through the sensor and metabolic models.")
        st.plotly_chart(wiring_plot(),width="stretch")
        wiring = pd.DataFrame([
            ("TCS34725","VIN","5V"),("TCS34725","GND","GND"),("TCS34725","SDA","A4 / SDA"),
            ("TCS34725","SCL","A5 / SCL"),("pH interface","VCC","5V"),
            ("pH interface","GND","GND"),("pH interface","AO","A0"),
            ("DS18B20","VDD","5V"),("DS18B20","GND","GND"),("DS18B20","DATA","D2"),
            ("4.7 kΩ resistor","one end","D2 / DATA"),("4.7 kΩ resistor","other end","5V")
        ],columns=["Module","Pin","Uno connection"])
        st.dataframe(wiring,hide_index=True,width="stretch")
        firmware = PROJECT_DIR/"rcds_arduino_uno.ino"
        if firmware.exists():
            st.download_button("Download Arduino Uno firmware (.ino)",firmware.read_bytes(),
                               file_name=firmware.name,mime="text/plain",type="primary")
            with st.expander("View firmware"): st.code(firmware.read_text(),language="cpp")
        st.caption("Wiring matches the included sketch; verify voltage/logic-level requirements for your breakout-board revision.")
    st.divider()
    st.caption("Research prototype · illustrative pathway units · not for diagnosis or clinical decisions")


if __name__ == "__main__":
    main()
