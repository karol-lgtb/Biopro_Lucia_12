import io, json, zipfile, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="BioPro | Ensayo de cochinilla", page_icon="🍍", layout="wide")

# ---------- Diseño fijo confirmado ----------
GRUPOS = ["T1 MOV", "T2", "T3", "Control"]
SEMANAS = [1, 2, 3, 4]
PLANTAS = list(range(1, 11))
VARS = ["pH", "CE", "Brix"]
PALETTE = {"T1 MOV":"#287271","T2":"#2A9D8F","T3":"#E9C46A","Control":"#6C757D"}

st.markdown("""
<style>
.block-container{max-width:1180px;padding-top:2rem;padding-bottom:4rem}
.hero{padding:1.4rem 1.6rem;border-radius:18px;background:linear-gradient(120deg,#153B50,#287271);color:white;margin-bottom:1rem}
.hero h1{margin:0;font-size:2rem}.hero p{margin:.35rem 0 0;opacity:.9}
.card{border:1px solid #e8ecef;border-radius:14px;padding:1rem;background:white}
.small{color:#65727c;font-size:.9rem}
div[data-testid="stMetric"]{border:1px solid #e8ecef;padding:.8rem;border-radius:14px}
</style>
<div class="hero"><h1>BioPro · Análisis de ensayo</h1>
<p>Control preventivo de cochinilla · Lote Lucía 12 · Seguimiento de pH, CE y °Brix</p></div>
""", unsafe_allow_html=True)

def plantilla():
    return pd.DataFrame(
        [[s,g,p,np.nan,np.nan,np.nan] for s in SEMANAS for g in GRUPOS for p in PLANTAS],
        columns=["Semana","Grupo","Planta","pH","CE","Brix"]
    )

def normalizar(df):
    req = ["Semana","Grupo","Planta","pH","CE","Brix"]
    if not all(c in df.columns for c in req):
        raise ValueError("La hoja DATOS debe contener: Semana, Grupo, Planta, pH, CE y Brix.")
    x=df[req].copy()
    x["Grupo"]=x["Grupo"].astype(str).str.strip()
    for c in ["Semana","Planta","pH","CE","Brix"]:
        if not pd.api.types.is_numeric_dtype(x[c]):
            x[c]=x[c].astype(str).str.replace(",",".",regex=False)
        x[c]=pd.to_numeric(x[c],errors="coerce")
    return x

def validar(x):
    errores=[]; alertas=[]; estado=[]
    extras=sorted(set(x["Grupo"].dropna())-set(GRUPOS))
    if extras: errores.append("Grupos no autorizados: "+", ".join(extras))
    if x["Semana"].dropna().isin(SEMANAS).all() is False: errores.append("Hay semanas fuera de 1–4.")
    if x["Planta"].dropna().isin(PLANTAS).all() is False: errores.append("Hay plantas fuera de 1–10.")
    d=x.duplicated(["Semana","Grupo","Planta"],keep=False)
    if d.any(): errores.append("Hay registros Semana–Grupo–Planta duplicados.")
    completas=[]
    for s in SEMANAS:
        sok=True
        for g in GRUPOS:
            q=x[(x.Semana==s)&(x.Grupo==g)]
            n=int(q[VARS].notna().all(axis=1).sum())
            ok=(len(q)==10 and n==10)
            estado.append([s,g,len(q),n,"Completo" if ok else "Pendiente"])
            sok &= ok
        if sok: completas.append(s)
    # Solo banderas, nunca eliminación automática.
    for v in VARS:
        vals=x[v].dropna()
        if len(vals)>=8:
            q1,q3=vals.quantile([.25,.75]); iqr=q3-q1
            lo,hi=q1-3*iqr,q3+3*iqr
            sospe=x[x[v].notna() & ((x[v]<lo)|(x[v]>hi))]
            for _,r in sospe.iterrows():
                alertas.append(f"Revisar {v}: Semana {int(r.Semana)}, {r.Grupo}, planta {int(r.Planta)} = {r[v]:g}. No fue eliminado.")
    return errores, alertas, pd.DataFrame(estado,columns=["Semana","Grupo","Filas","Plantas completas","Estado"]), completas

def descriptivos(x,semanas):
    out=[]
    for s in semanas:
        for g in GRUPOS:
            q=x[(x.Semana==s)&(x.Grupo==g)]
            for v in VARS:
                z=q[v].dropna().astype(float); n=len(z)
                m=z.mean(); sd=z.std(ddof=1)
                h=stats.t.ppf(.975,n-1)*sd/np.sqrt(n) if n>1 else np.nan
                out.append([s,g,v,n,m,sd,np.median(z),m-h,m+h])
    return pd.DataFrame(out,columns=["Semana","Grupo","Variable","n","Media","DE","Mediana","IC95_inf","IC95_sup"])

def pruebas_semana(x,semanas):
    out=[]
    for s in semanas:
        q=x[x.Semana==s]
        for v in VARS:
            grupos=[q.loc[q.Grupo==g,v].dropna().values for g in GRUPOS]
            # Supuestos informativos; no se escoge una prueba por conveniencia.
            shapiro=[stats.shapiro(a).pvalue if len(a)>=3 else np.nan for a in grupos]
            lev_p=stats.levene(*grupos,center="median").pvalue
            normal=all(p>.05 for p in shapiro if not np.isnan(p))
            if normal and lev_p>.05:
                stat,p=stats.f_oneway(*grupos); prueba="ANOVA de una vía"
            else:
                stat,p=stats.kruskal(*grupos); prueba="Kruskal–Wallis"
            out.append([s,v,prueba,stat,p,lev_p,min(shapiro)])
    return pd.DataFrame(out,columns=["Semana","Variable","Prueba","Estadístico","p","Levene_p","Shapiro_min_p"])

def longitudinal(x,semanas):
    if len(semanas)<2:
        return pd.DataFrame(), "No habilitado: se requieren al menos dos semanas individuales completas."
    try:
        import statsmodels.formula.api as smf
        d=x[x.Semana.isin(semanas)].copy()
        d["ID"]=d["Grupo"]+"_P"+d["Planta"].astype(int).astype(str)
        rows=[]
        for v in VARS:
            full=smf.mixedlm(f'Q("{v}") ~ C(Semana)*C(Grupo, Treatment(reference="Control"))',d,groups=d["ID"]).fit(reml=False,method="lbfgs",disp=False)
            add=smf.mixedlm(f'Q("{v}") ~ C(Semana)+C(Grupo, Treatment(reference="Control"))',d,groups=d["ID"]).fit(reml=False,method="lbfgs",disp=False)
            lr=2*(full.llf-add.llf); gl=max(1,int(full.df_modelwc-add.df_modelwc)); p=stats.chi2.sf(lr,gl)
            rows.append([v,lr,gl,p,bool(p<.05),bool(full.converged)])
        return pd.DataFrame(rows,columns=["Variable","LR interacción","gl","p interacción","Significativa","Convergió"]), "Habilitado"
    except Exception as e:
        return pd.DataFrame(), f"Modelo no estimable: {e}"

def fig_variable(desc,v):
    fig,ax=plt.subplots(figsize=(8.2,4.5))
    q=desc[desc.Variable==v]
    for g in GRUPOS:
        z=q[q.Grupo==g].sort_values("Semana")
        if len(z):
            ax.errorbar(z.Semana,z.Media,yerr=[z.Media-z.IC95_inf,z.IC95_sup-z.Media],
                        marker="o",capsize=4,label=g,color=PALETTE[g])
    ax.set_xlabel("Semana"); ax.set_ylabel(v if v!="Brix" else "°Brix")
    ax.set_xticks(SEMANAS); ax.grid(alpha=.18); ax.legend(frameon=False,ncol=4,fontsize=8)
    fig.tight_layout()
    return fig

def fig_bytes(fig):
    b=io.BytesIO(); fig.savefig(b,format="png",dpi=220,bbox_inches="tight"); b.seek(0); return b.getvalue()

def latex(desc,pruebas,longi,long_estado,semanas):
    def esc(s): return str(s).replace("&",r"\&").replace("%",r"\%")
    L=[r"""\documentclass[11pt]{article}
\usepackage[utf8]{inputenc}
\usepackage[spanish]{babel}
\usepackage[a4paper,margin=1.8cm]{geometry}
\usepackage{booktabs,graphicx,xcolor,array}
\definecolor{azul}{RGB}{21,59,80}
\definecolor{verde}{RGB}{40,114,113}
\setlength{\parindent}{0pt}
\begin{document}
{\LARGE\bfseries\color{azul} Ensayo de control de cochinilla}\\[2mm]
\textbf{Lote:} Lucía 12 \hfill \textbf{Efecto:} Preventivo\\
\textbf{Grupos:} T1 MOV, T2, T3 y Control\\[3mm]
\hrule
\vspace{4mm}
\section*{\color{azul}Estado del análisis}"""]
    L.append("Semanas con datos individuales completos: "+(", ".join(map(str,semanas)) if semanas else "ninguna")+r".\\")
    L.append(esc(long_estado)+r"\\")
    L.append(r"\section*{\color{azul}Resumen descriptivo}")
    L.append(r"\begin{tabular}{lllrrr}\toprule Semana & Grupo & Variable & $n$ & Media & DE\\\midrule")
    for _,r in desc.iterrows():
        L.append(f"{int(r.Semana)} & {esc(r.Grupo)} & {esc(r.Variable)} & {int(r.n)} & {r.Media:.3f} & {r.DE:.3f}\\\\")
    L.append(r"\bottomrule\end{tabular}")
    L.append(r"\section*{\color{azul}Comparaciones entre grupos}")
    if len(pruebas):
        L.append(r"\begin{tabular}{lllr}\toprule Semana & Variable & Prueba & $p$\\\midrule")
        for _,r in pruebas.iterrows():
            L.append(f"{int(r.Semana)} & {esc(r.Variable)} & {esc(r.Prueba)} & {r.p:.4f}\\\\")
        L.append(r"\bottomrule\end{tabular}")
    L.append(r"\section*{\color{azul}Evolución}")
    for v in VARS:
        L.append(rf"\begin{{center}}\includegraphics[width=.86\textwidth]{{{v}_evolucion.png}}\end{{center}}")
    L.append(r"\section*{\color{azul}Criterio de interpretación}")
    L.append(r"Los valores faltantes no se imputaron. Las observaciones marcadas para revisión no se eliminaron automáticamente. Una diferencia temporal dentro de un tratamiento no se atribuye al producto sin evidencia de una evolución diferente respecto al Control. Las asociaciones entre variables no se interpretan como causalidad.")
    L.append(r"\end{document}")
    return "\n".join(L)

# ---------- Navegación ----------
with st.expander("¿Cómo utilizar esta herramienta?", expanded=False):
    st.markdown("""
    **1. Datos:** cargue el Excel del ensayo.  
    **2. Validación:** revise cualquier alerta antes de continuar.  
    **3. Resultados:** consulte la evolución de pH, CE y °Brix y los análisis habilitados.  
    **4. Informe:** descargue el paquete listo para documentar los resultados.

    La herramienta **no completa, corrige ni elimina mediciones automáticamente**.
    """)

entrada, resultados, informe = st.tabs(["1 · Datos", "2 · Resultados", "3 · Informe"])

with entrada:
    c1,c2=st.columns([1,1])
    with c1:
        st.subheader("Cargar datos")
        f=st.file_uploader("Excel del ensayo",type=["xlsx"],help="Debe contener una hoja llamada DATOS.")
        st.caption("La aplicación nunca completa mediciones faltantes.")
    with c2:
        st.subheader("O usar plantilla")
        tpl=plantilla()
        bio=io.BytesIO()
        with pd.ExcelWriter(bio,engine="openpyxl") as w: tpl.to_excel(w,sheet_name="DATOS",index=False)
        st.download_button("Descargar plantilla vacía",bio.getvalue(),"plantilla_ensayo_cochinilla.xlsx")
        st.caption("160 filas fijas: 4 semanas × 4 grupos × 10 plantas.")

    if f:
        try:
            st.session_state["datos"]=normalizar(pd.read_excel(f,sheet_name="DATOS"))
            st.success("Archivo leído correctamente.")
        except Exception as e: st.error(str(e))

    if "datos" in st.session_state:
        x=st.session_state["datos"]
        err,alert,estado,completas=validar(x)
        st.subheader("Control de calidad")
        a,b,c,d=st.columns(4)
        a.metric("Grupos","4 / 4" if not any("Grupos" in z for z in err) else "Revisar")
        b.metric("Semanas completas",len(completas))
        c.metric("Plantas por grupo","10")
        d.metric("Alertas",len(alert))
        if err:
            for e in err: st.error(e)
        else: st.success("Estructura válida. No se detectaron errores que impidan el análisis.")
        for a0 in alert: st.warning(a0)
        st.dataframe(estado,hide_index=True,use_container_width=True)

with resultados:
    if "datos" not in st.session_state:
        st.info("Primero cargue un archivo en la pestaña Datos.")
    else:
        x=st.session_state["datos"]; err,alert,estado,completas=validar(x)
        if err: st.error("Corrija los errores estructurales antes de analizar.")
        elif not completas: st.warning("Todavía no hay una semana individual completa.")
        else:
            desc=descriptivos(x,completas); pruebas=pruebas_semana(x,completas)
            longi,long_estado=longitudinal(x,completas)
            st.subheader("Resumen ejecutivo")
            m1,m2,m3=st.columns(3)
            m1.metric("Semanas analizables",len(completas))
            m2.metric("Observaciones completas",int(x[x.Semana.isin(completas)][VARS].notna().all(axis=1).sum()))
            m3.metric("Longitudinal","Sí" if len(completas)>=2 else "Aún no")
            st.caption("Las pruebas se seleccionan mediante reglas predefinidas de supuestos; no se elige el resultado con menor p.")
            for v in VARS:
                st.markdown(f"### {'°Brix' if v=='Brix' else v}")
                fig=fig_variable(desc,v); st.pyplot(fig,use_container_width=False); plt.close(fig)
            with st.expander("Ver estadística descriptiva"):
                st.dataframe(desc,hide_index=True,use_container_width=True)
            with st.expander("Ver pruebas entre grupos"):
                st.dataframe(pruebas,hide_index=True,use_container_width=True)
            with st.expander("Ver análisis longitudinal"):
                st.write(long_estado)
                if len(longi): st.dataframe(longi,hide_index=True,use_container_width=True)

with informe:
    if "datos" not in st.session_state:
        st.info("Cargue los datos primero.")
    else:
        x=st.session_state["datos"]; err,alert,estado,completas=validar(x)
        if err or not completas:
            st.warning("El informe no se habilita hasta que exista al menos una semana completa y la estructura sea válida.")
        else:
            desc=descriptivos(x,completas); pruebas=pruebas_semana(x,completas)
            longi,long_estado=longitudinal(x,completas)
            tex=latex(desc,pruebas,longi,long_estado,completas)
            figs={}
            for v in VARS:
                fig=fig_variable(desc,v); figs[f"{v}_evolucion.png"]=fig_bytes(fig); plt.close(fig)
            paquete=io.BytesIO()
            with zipfile.ZipFile(paquete,"w",zipfile.ZIP_DEFLATED) as z:
                z.writestr("informe_ensayo.tex",tex)
                for n,b in figs.items(): z.writestr(n,b)
                z.writestr("descriptivos.csv",desc.to_csv(index=False))
                z.writestr("pruebas_entre_grupos.csv",pruebas.to_csv(index=False))
                z.writestr("validacion.csv",estado.to_csv(index=False))
                if len(longi): z.writestr("modelo_longitudinal.csv",longi.to_csv(index=False))
            st.subheader("Informe listo para Overleaf")
            st.write("Incluye el código LaTeX, las tres figuras y las tablas de respaldo.")
            st.download_button("Descargar paquete completo",paquete.getvalue(),
                               "informe_ensayo_cochinilla.zip","application/zip",type="primary")
            with st.expander("Vista previa del código LaTeX"):
                st.code(tex,language="latex")

st.divider()
st.caption("BioPro · Herramienta interna de apoyo analítico · Los resultados deben interpretarse dentro del diseño experimental y el contexto agronómico.")
