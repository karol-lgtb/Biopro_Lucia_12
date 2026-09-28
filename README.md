# App final — Ensayo de control de cochinilla

## Qué hace
- Recibe el Excel del ensayo.
- Valida T1 MOV, T2, T3, Control; semanas 1–4; plantas 1–10.
- No imputa datos faltantes.
- Señala valores sospechosos sin eliminarlos.
- Genera descriptivos e IC95%.
- Selecciona ANOVA o Kruskal–Wallis mediante reglas explícitas de supuestos.
- Habilita modelo mixto longitudinal cuando hay al menos dos semanas completas.
- Genera gráficos y un paquete LaTeX listo para Overleaf.

## Publicación
Suba `app.py` y `requirements.txt` a un repositorio de GitHub y despliegue `app.py`
en Streamlit Community Cloud.

## Nota metodológica
La identidad Planta debe mantenerse entre semanas para interpretar el análisis
longitudinal como medidas repetidas.
