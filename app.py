# -*- coding: utf-8 -*-
"""HydroRisk: rational method, Manning and conditional Monte Carlo."""
import copy
import io
import json
import math
from graphlib import TopologicalSorter, CycleError
from xml.sax.saxutils import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats
import streamlit as st

BASIN_COLS = ['id', 'troncon', 'occupation', 'surface_ha', 'coefficient']
PIPE_COLS = ['id', 'aval', 'diametre_mm', 'pente_pct', 'manning_n']
LIMITS = ('Pluie uniforme, pointes simultanées, pas de routage ni de stockage. '
          'Capacité théorique en régime uniforme ; pas de calcul de mise en charge, '
          'de hauteur ou de zone inondée. Probabilités conditionnelles au scénario, '
          'sans fréquence annuelle ni validation réglementaire.')
SOURCES = {'FHWA HDS-2': 'https://www.fhwa.dot.gov/engineering/hydraulics/pubs/hif24007.pdf',
           'FHWA HEC-22': 'https://www.fhwa.dot.gov/engineering/hydraulics/pubs/hif24006.pdf'}


def example():
    return dict(basins=[dict(zip(BASIN_COLS, r)) for r in [
        ('B1', 'T1', 'Toitures', .8, .9), ('B2', 'T2', 'Voirie', .7, .8),
        ('B3', 'T3', 'Espaces verts', .5, .25)]],
        pipes=[dict(zip(PIPE_COLS, r)) for r in [
            ('T1', 'T3', 400., .6, .013), ('T2', 'T3', 400., .5, .013),
            ('T3', '', 500., .4, .013)]],
        intensity=100., duration=45., tc=20., cv_i=.2, cv_c=.1, n=10000, seed=42,
        label='Exemple fictif pédagogique')


def number(value, low, high, label):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{label} : nombre attendu.')
    try:
        v = float(value)
    except (ValueError, TypeError):
        raise ValueError(f'{label} : nombre attendu.') from None
    if not math.isfinite(v) or not low <= v <= high:
        raise ValueError(f'{label} : valeur entre {low} et {high}.')
    return v


def clean(frame, columns):
    if list(frame.columns) != columns:
        raise ValueError('Colonnes attendues, dans cet ordre : ' + ', '.join(columns))
    if frame.empty or len(frame) > 100:
        raise ValueError('Chaque tableau doit contenir de 1 à 100 lignes.')
    frame = frame.copy()
    for col in columns:
        if col in ('id', 'troncon', 'occupation', 'aval'):
            frame[col] = frame[col].fillna('').astype(str).str.strip()
            if frame[col].str.len().max() > 60:
                raise ValueError(f'{col} : 60 caractères maximum.')
        else:
            frame[col] = pd.to_numeric(frame[col], errors='coerce')
            if not np.isfinite(frame[col]).all():
                raise ValueError(f'{col} : cellule vide ou nombre invalide.')
    if (frame.id == '').any() or frame.id.duplicated().any():
        raise ValueError('Les identifiants doivent être renseignés et uniques.')
    return frame


def validate(config):
    if not isinstance(config, dict):
        raise ValueError('Projet JSON : objet attendu.')
    required={'basins','pipes','intensity','duration','tc','cv_i','cv_c','n','seed'}
    if not required <= config.keys():
        raise ValueError('Projet incomplet : exporter un projet JSON HydroRisk complet.')
    for key,columns in [('basins',BASIN_COLS),('pipes',PIPE_COLS)]:
        if not isinstance(config[key],list) or any(not isinstance(row,dict) or set(row)!=set(columns) for row in config[key]):
            raise ValueError(f'{key} : colonnes manquantes ou supplémentaires.')
    for key in required - {'basins','pipes'}:
        if not isinstance(config[key], (int,float,np.number)) or isinstance(config[key],bool):
            raise ValueError(f'{key} : nombre JSON attendu, sans guillemets.')
    for rows,fields in [(config['basins'],('surface_ha','coefficient')),
                        (config['pipes'],('diametre_mm','pente_pct','manning_n'))]:
        for row in rows:
            if any(not isinstance(row[k],(int,float,np.number)) or isinstance(row[k],bool) for k in fields):
                raise ValueError('Les valeurs numériques du projet JSON doivent être des nombres.')
    if 'label' in config and (not isinstance(config['label'],str) or len(config['label'])>150):
        raise ValueError('Libellé de projet : texte de 150 caractères maximum.')
    b = clean(pd.DataFrame(config['basins'], columns=BASIN_COLS), BASIN_COLS)
    p = clean(pd.DataFrame(config['pipes'], columns=PIPE_COLS), PIPE_COLS)
    for v in b.surface_ha: number(v, .0001, 10000, 'Surface (ha)')
    for v in b.coefficient: number(v, 0, 1, 'Coefficient C')
    for v in p.diametre_mm: number(v, 50, 5000, 'Diamètre (mm)')
    for v in p.pente_pct: number(v, .001, 30, 'Pente (%)')
    for v in p.manning_n: number(v, .005, .1, 'Manning n')
    ids = set(p.id)
    if not set(b.troncon) <= ids or not set(p.aval) <= ids | {''}:
        raise ValueError('Un bassin ou un tronçon fait référence à un tronçon inexistant.')
    outlets = p.loc[p.aval == '', 'id'].tolist()
    if len(outlets) != 1:
        raise ValueError('Prévoir exactement un exutoire : aval vide pour un seul tronçon.')
    predecessors = {key: set() for key in p.id}
    for row in p.itertuples():
        if row.aval: predecessors[row.aval].add(row.id)
    try:
        order = list(TopologicalSorter(predecessors).static_order())
    except CycleError:
        raise ValueError('Le réseau contient un cycle. Les branches doivent converger vers l’exutoire.') from None
    for key, lo, hi in [('intensity', .01, 2000), ('duration', .1, 1440), ('tc', .1, 1440),
                        ('cv_i', 0, 1), ('cv_c', 0, .8), ('n', 1000, 50000), ('seed', 0, 2147483647)]:
        number(config[key], lo, hi, key)
    if int(config['n']) != config['n'] or int(config['seed']) != config['seed']:
        raise ValueError('Nombre de tirages et graine : entiers attendus.')
    # A beta law is only defined for variance strictly below mean*(1-mean).
    for c in b.coefficient:
        if 0 < c < 1 and config['cv_c'] and (c * config['cv_c']) ** 2 >= c * (1-c):
            raise ValueError('CV de C trop élevé pour au moins un coefficient : réduire ce CV.')
    return b, p, predecessors, order, outlets[0]


def manning(diameter_mm, slope_pct, n):
    d = np.asarray(diameter_mm) / 1000
    return (math.pi*d*d/4) * (d/4)**(2/3) * np.sqrt(np.asarray(slope_pct)/100) / np.asarray(n)


def calculate(config, intensity=None, coefficients=None):
    b, p, upstream, order, outlet = validate(config)
    intensity = config['intensity'] if intensity is None else intensity
    coefficients = b.coefficient.to_numpy() if coefficients is None else coefficients
    local = {key: 0. for key in order}
    for j, row in enumerate(b.itertuples()):
        local[row.troncon] = local[row.troncon] + coefficients[j]*intensity*row.surface_ha/360
    flows = {}
    for key in order:
        flows[key] = local[key] + sum(flows[k] for k in upstream[key])
    capacity = dict(zip(p.id, manning(p.diametre_mm, p.pente_pct, p.manning_n)))
    return b, p, outlet, flows, capacity


def results(config):
    b, p, outlet, flows, caps = calculate(config)
    table = p.copy()
    table['debit_m3s'] = table.id.map(flows)
    table['capacite_m3s'] = table.id.map(caps)
    table['utilisation_pct'] = table.debit_m3s / table.capacite_m3s * 100
    table['depassement'] = table.debit_m3s > table.capacite_m3s
    return b, table, outlet


def confidence(mask):
    count, total = int(np.count_nonzero(mask)), len(mask)
    ci = stats.binomtest(count, total).proportion_ci(method='exact')
    return dict(count=count, total=total, p=count/total, low=float(ci.low), high=float(ci.high))


def simulate(config):
    b, _, _, _, _ = validate(config)
    rng = np.random.default_rng(int(config['seed'])); n = int(config['n'])
    z = rng.standard_normal(n); s = np.sqrt(np.log1p(config['cv_i']**2))
    rain = config['intensity']*np.exp(s*z-s*s/2)
    cs = []
    for c in b.coefficient:
        u = rng.uniform(np.finfo(float).eps, 1-np.finfo(float).eps, n)
        if c in (0, 1) or config['cv_c'] == 0:
            cs.append(np.full(n, c))
        else:
            variance = (c*config['cv_c'])**2
            k = c*(1-c)/variance-1
            cs.append(stats.beta.ppf(u, c*k, (1-c)*k))
    _, p, outlet, flows, caps = calculate(config, rain, cs)
    risks = {key: confidence(flows[key] > caps[key]) for key in p.id}
    ratios = np.stack([flows[key]/caps[key] for key in p.id])
    worst = np.max(ratios, axis=0)
    system = confidence(worst > 1)
    coeff = np.average(np.array(cs), axis=0, weights=b.surface_ha)
    sensitivity = []
    for label, values in [('Pluie', rain), ('C pondéré', coeff)]:
        if np.ptp(values) > 1e-12 and np.ptp(worst) > 1e-12:
            sensitivity.append((label, float(stats.spearmanr(values, worst).statistic)))
    frame = pd.DataFrame({'intensite_mmh': rain, 'C_pondere': coeff, 'debit_exutoire_m3s': flows[outlet],
                          'ratio_max_reseau': worst, 'depassement_reseau': worst > 1})
    for key in p.id: frame[f'Q_{key}_m3s'] = flows[key]
    return dict(frame=frame, risks=risks, system=system, outlet=outlet, capacity=caps[outlet], sensitivity=sensitivity)


def excel_bytes(sheets):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for name, frame in sheets.items():
            # Export text as literal cells, not spreadsheet formulas.
            frame.to_excel(writer, sheet_name=name, index=False)
            for row in writer.sheets[name].iter_rows():
                for cell in row:
                    if cell.data_type == 'f': cell.data_type = 's'
    return output.getvalue()


def csv_bytes(frame):
    frame = frame.copy()
    for col in frame:
        if pd.api.types.is_string_dtype(frame[col]) or frame[col].dtype == object:
            frame[col] = frame[col].map(lambda x: "'"+x if isinstance(x, str) and x.startswith(('=', '+', '-', '@')) else x)
    return frame.to_csv(index=False).encode('utf-8-sig')


def pdf_bytes(config, sim=None):
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    output = io.BytesIO(); styles = getSampleStyleSheet(); story = []
    def para(text): return Paragraph(escape(str(text)), styles['BodyText'])
    def add(text): story.extend([para(text), Spacer(1, 10)])
    def tab(rows, widths):
        t = Table([[para(v) for v in r] for r in rows], colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#d9f1ff')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),6)]));story.extend([t,Spacer(1,12)])
    story.append(Paragraph('HydroRisk - Rapport pédagogique', styles['Title']))
    add(config.get('label', 'Données utilisateur'))
    add(f"Pluie : {config['intensity']:g} mm/h ; durée : {config['duration']:g} min ; temps de concentration global : {config['tc']:g} min.")
    if config['duration'] < config['tc']: add('HYPOTHÈSE NON SATISFAITE : durée inférieure au temps de concentration. Débits à titre indicatif uniquement.')
    b, r, outlet = results(config)
    tab([['Bassin / occupation','Tronçon','Surface (ha)','C']] + [[f'{v.id} / {v.occupation}',v.troncon,f'{v.surface_ha:g}',f'{v.coefficient:g}'] for v in b.itertuples()], [200,80,80,100])
    tab([['Tronçon / aval','D (mm) / pente (%) / n','Q / capacité (m³/s)']] + [[f'{v.id} / {v.aval or "exutoire"}',f'{v.diametre_mm:g} / {v.pente_pct:g} / {v.manning_n:g}',f'{v.debit_m3s:.5f} / {v.capacite_m3s:.5f}'] for v in r.itertuples()], [120,170,170])
    add('Q = C × i × A / 360 ; capacité = S × Rh^(2/3) × racine(pente) / n ; Rh = D/4. Unités SI pour Manning, pente en m/m.')
    if sim:
        add(f"Simulation : n = {config['n']} ; graine = {config['seed']} ; CV pluie = {config['cv_i']:.1%} ; CV C = {config['cv_c']:.1%}.")
        add('Pluie lognormale commune à tous les bassins ; coefficients bêta indépendants entre bassins et indépendants de la pluie. C = 0 ou 1 est fixé même si un CV est demandé.')
        risk = sim['system'];add(f"Dépassement d’au moins un tronçon : {risk['p']:.2%} ; intervalle exact à 95 % [{risk['low']:.2%} ; {risk['high']:.2%}].")
        q = sim['frame'].debit_exutoire_m3s.quantile([.05,.5,.95]);add(f'Q exutoire P05 / médiane / P95 : {q.iloc[0]:.5f} / {q.iloc[1]:.5f} / {q.iloc[2]:.5f} m³/s.')
    add(LIMITS)
    for name,url in SOURCES.items(): add(f'{name} : {url}')
    SimpleDocTemplate(output, leftMargin=48, rightMargin=48).build(story)
    return output.getvalue()


STYLE = '''<style>
.stApp {background:#f1f7fc;color:#193852}
[data-testid="stMainBlockContainer"] {max-width:1450px;padding-top:4.2rem}
[data-testid="stSidebar"] {background:#065d87}
[data-testid="stSidebar"] h2,[data-testid="stSidebar"] h3,[data-testid="stSidebar"] label {color:#effff9!important}
[data-testid="stSidebar"] button {background:#0977a6;color:white;border-color:#50afd0;min-height:42px}
[data-testid="stSidebar"] button:hover {background:#095279;border-color:#a8e6fc}
[data-testid="stSidebar"] button:disabled {background:#386e86;color:#c6dbe4;border-color:#5d8ca0}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color:#d7effa}
.hr-banner {background:#d9f1ff;border-left:7px solid #f0b426;border-bottom:2px solid #a5d4ee;padding:24px 28px;margin-bottom:20px}
.hr-banner h1 {color:#064f7c;font-size:2rem;margin:0;padding:0;line-height:1.3}
.hr-banner p {color:#2b617f;margin:8px 0 0}
.hr-banner .eyebrow {color:#116d99;font-size:12px;font-weight:bold;margin:0 0 7px}
[class*="st-key-kpi_"] {border-radius:6px;padding:16px;min-height:130px}
.st-key-kpi_0 {background:#087aac}.st-key-kpi_1 {background:#daf3fc;border-top:5px solid #19a6c9}
.st-key-kpi_2 {background:#ffedba;border-top:5px solid #e7aa19}.st-key-kpi_3 {background:#ffe1d8;border-top:5px solid #e37b5d}
[class*="st-key-kpi_"] [data-testid="stMetricValue"], [class*="st-key-kpi_"] [data-testid="stMetricLabel"] {color:white!important}
.st-key-kpi_1 [data-testid="stMetricValue"], .st-key-kpi_1 [data-testid="stMetricLabel"] {color:#075575!important}
.st-key-kpi_2 [data-testid="stMetricValue"], .st-key-kpi_2 [data-testid="stMetricLabel"] {color:#704900!important}
.st-key-kpi_3 [data-testid="stMetricValue"], .st-key-kpi_3 [data-testid="stMetricLabel"] {color:#8d361f!important}
[data-testid="stMetricValue"] {font-size:1.65rem!important;white-space:normal}
button {border-radius:6px!important} button p {white-space:normal}
[data-testid="stBaseButton-primary"], [data-testid="stFormSubmitButton"] button {background:#087aac;color:white;border-color:#087aac}
[data-testid="stForm"] {background:#f9fcff;border-top:4px solid #19a6c9;border-radius:6px}
h1,h2,h3,p {letter-spacing:0!important}
@media(max-width:640px) {
 [data-testid="stMainBlockContainer"] {padding:4.2rem 1rem 1.5rem}
 [data-testid="stColumn"] {width:100%;flex:1 1 100%;min-width:0}
 .hr-banner {padding:20px 18px}.hr-banner h1 {font-size:1.8rem}
 input {font-size:16px!important}
}
</style>'''


def chart(fig, key):
    fig.update_layout(template='plotly_white', paper_bgcolor='rgba(0,0,0,0)', height=365,
                      margin=dict(l=25,r=20,t=40,b=60), font=dict(color='#234b65'),
                      legend=dict(orientation='h', y=-.25), colorway=['#087aac','#e5aa24','#e37b5d','#19a6c9'])
    fig.update_xaxes(automargin=True)
    fig.update_yaxes(automargin=True)
    st.plotly_chart(fig, key=key, width='stretch', config={'displayModeBar':False})


def network_chart(config):
    b,p,up,order,outlet=validate(config); _,r,_=results(config)
    levels={}
    for key in reversed(order):
        dest=p.set_index('id').loc[key,'aval'];levels[key]=levels[dest]+1 if dest else 0
    pos={}
    for lev in set(levels.values()):
        nodes=[k for k in order if levels[k]==lev]
        for idx,k in enumerate(nodes):pos[k]=((idx+1)/(len(nodes)+1),lev)
    fig=go.Figure()
    for row in r.itertuples():
        x,y=pos[row.id];dest=pos[row.aval] if row.aval else (x,y-.65)
        fig.add_trace(go.Scatter(x=[x,dest[0]],y=[y,dest[1]],mode='lines',line=dict(width=6,color='#cd593c' if row.depassement else '#159bc1'),showlegend=False,hoverinfo='skip'))
        fig.add_annotation(x=dest[0],y=dest[1],ax=x,ay=y,axref='x',ayref='y',text='',showarrow=True,arrowhead=2,arrowwidth=2)
    fig.add_trace(go.Scatter(x=[pos[k][0] for k in p.id],y=[pos[k][1] for k in p.id],mode='markers+text',text=p.id,
                            textposition='top center',marker=dict(size=20,color='#2675b1'),customdata=[[f'{v.debit_m3s:.3f}',f'{v.capacite_m3s:.3f}',f'{b.loc[b.troncon==v.id,"surface_ha"].sum():g}'] for v in r.itertuples()],
                            hovertemplate='%{text}<br>Q %{customdata[0]} m³/s<br>Capacité %{customdata[1]} m³/s<br>Surface locale %{customdata[2]} ha<extra></extra>',showlegend=False))
    fig.add_annotation(x=pos[outlet][0],y=-.65,text='Exutoire',showarrow=False,yshift=-15)
    fig.update_xaxes(visible=False,range=[0,1]);fig.update_yaxes(visible=False,range=[-1,max(levels.values())+.6])
    return fig


VIEWS={'Tableau de bord':'dashboard','Bassin versant':'landscape','Pluie et ruissellement':'rainy',
       'Réseau simplifié':'account_tree','Simulation':'query_stats','Comparaison':'compare_arrows',
       'Données et exports':'download','Méthode et tests':'functions'}


def navigate(view): st.session_state.view=view


def load_example():
    st.session_state.config=example()
    st.session_state.editor_revision=st.session_state.get('editor_revision',0)+1
    for key in ('simulation_result','comparison_export'):st.session_state.pop(key,None)


def commit_config(config):
    validate(config)
    st.session_state.config=config
    st.session_state.editor_revision=st.session_state.get('editor_revision',0)+1
    st.session_state.pop('simulation_result',None)


def main():
    st.set_page_config(page_title='HydroRisk',page_icon=':material/water_drop:',layout='wide')
    st.html(STYLE)
    st.html('<header class="hr-banner"><p class="eyebrow">HYDROLOGIE URBAINE / LABORATOIRE</p><h1>HydroRisk</h1><p>Ruissellement / Drainage / Incertitudes</p></header>')
    if 'config' not in st.session_state:
        c=example();c.update(basins=[],pipes=[],label='Projet utilisateur');st.session_state.config=c
    config=copy.deepcopy(st.session_state.config)
    with st.sidebar:
        st.subheader('HydroRisk')
        for view,icon in VIEWS.items():
            st.button(view,icon=f':material/{icon}:',on_click=navigate,args=(view,),width='stretch',key=f'nav_{icon}')
        st.divider()
        replace_ok=st.checkbox('Autoriser le remplacement par l’exemple fictif',key='replace_example')
        st.button('Charger l’exemple fictif',icon=':material/science:',disabled=bool(config['basins']) and not replace_ok,on_click=load_example,width='stretch')
        st.caption('Modèle pédagogique • aucun compte requis')
    view=st.session_state.get('view','Tableau de bord');st.subheader(view)
    if not config['basins']:
        st.info('Aucune donnée chargée. Créez votre bassin ou chargez volontairement l’exemple fictif.')
        if view not in ('Bassin versant','Réseau simplifié','Données et exports','Méthode et tests'):
            for idx,(col,(name,value)) in enumerate(zip(st.columns(4),[('Débit','—'),('Capacité','—'),('Utilisation','—'),('Probabilité','—')])):
                with col,st.container(key=f'kpi_{idx}'):st.metric(name,value)
            st.button('Créer mon bassin',icon=':material/add:',on_click=navigate,args=('Bassin versant',),type='primary')
            return
    ready=bool(config['basins'] and config['pipes'])
    if ready:
        b,r,outlet=results(config);st.caption(config.get('label','Projet utilisateur'))
        if config['duration']<config['tc']:
            st.warning('Durée de pluie inférieure au temps de concentration global : hypothèse rationnelle non satisfaite. Résultats indicatifs, non valides comme pointes établies.')
    fingerprint=json.dumps(config,sort_keys=True,ensure_ascii=False)
    cached=st.session_state.get('simulation_result');sim=cached[1] if cached and cached[0]==fingerprint else None
    if view=='Bassin versant':
        st.caption('Chaque ligne représente une surface distincte, comptée une seule fois. Identifiant unique, surface en hectares, C entre 0 et 1.')
        base=pd.DataFrame(config['basins'],columns=BASIN_COLS)
        if base.empty:base=pd.DataFrame([['B1','T1','Surface à définir',2.,.6]],columns=BASIN_COLS)
        edited=st.data_editor(base,num_rows='dynamic',hide_index=True,width='stretch',key=f'basin_editor_{st.session_state.get("editor_revision",0)}')
        if st.button('Enregistrer le bassin',icon=':material/save:',type='primary'):
            draft=copy.deepcopy(config);draft['basins']=clean(edited,BASIN_COLS).to_dict('records')
            if not draft['pipes']:
                draft['pipes']=[dict(zip(PIPE_COLS,[key,'',500.,.5,.013])) for key in sorted(set(edited.troncon))]
            commit_config(draft);st.rerun()
        if ready:
            st.metric('Surface totale (ha)',f'{b.surface_ha.sum():.3f}')
            st.metric('C pondéré',f'{np.average(b.coefficient,weights=b.surface_ha):.3f}')
            chart(go.Figure(go.Pie(labels=b.occupation,values=b.surface_ha,hole=.55)), 'landuse')
    elif view=='Réseau simplifié':
        st.caption('aval = identifiant du tronçon suivant ; laisser vide uniquement à l’exutoire. Pente en %, diamètre intérieur en mm.')
        base=pd.DataFrame(config['pipes'],columns=PIPE_COLS)
        if base.empty:base=pd.DataFrame([['T1','',500.,.5,.013]],columns=PIPE_COLS)
        edited=st.data_editor(base,num_rows='dynamic',hide_index=True,width='stretch',key=f'pipe_editor_{st.session_state.get("editor_revision",0)}')
        if st.button('Enregistrer le réseau',icon=':material/save:',type='primary'):
            draft=copy.deepcopy(config);draft['pipes']=clean(edited,PIPE_COLS).to_dict('records')
            if not draft['basins']:raise ValueError('Créez d’abord un bassin ou importez un projet complet.')
            commit_config(draft);st.rerun()
        if ready:
            chart(network_chart(config),'network_full');st.dataframe(r,hide_index=True,width='stretch')
            st.caption('Les pointes sont additionnées sans routage, avec simultanéité supposée. Un dépassement ne constitue pas une prévision de hauteur d’inondation.')
    elif view=='Pluie et ruissellement' and ready:
        with st.form('rain'):
            i=st.number_input('Intensité uniforme (mm/h)',.01,2000.,float(config['intensity']))
            duration=st.number_input('Durée de pluie (min)',.1,1440.,float(config['duration']))
            tc=st.number_input('Temps de concentration global saisi (min)',.1,1440.,float(config['tc']))
            if st.form_submit_button('Appliquer la pluie',icon=':material/check:'):
                config.update(intensity=i,duration=duration,tc=tc);commit_config(config);st.rerun()
        st.metric('Débit théorique à l’exutoire (m³/s)',f'{r.set_index("id").loc[outlet,"debit_m3s"]:.4f}')
        chart(go.Figure(go.Bar(x=['Pluie','tc global'],y=[config['duration'],config['tc']],marker_color=['#087aac','#e5aa24'])).update_layout(yaxis_title='Durée (min)'), 'rain_duration')
        st.caption('Intensité saisie, sans période de retour attribuée. tc est fourni par l’utilisateur et inclut les trajets jusqu’à l’exutoire ; aucune courbe IDF locale n’est inventée. Aucun hydrogramme dynamique n’est calculé.')
    elif view=='Simulation' and ready:
        with st.form('uncertainty'):
            cvi=st.number_input('CV intensité (%)',0.,100.,float(config['cv_i']*100))
            cvc=st.number_input('CV des coefficients C (%)',0.,80.,float(config['cv_c']*100))
            n=st.selectbox('Nombre de tirages',[1000,5000,10000,25000,50000],index=[1000,5000,10000,25000,50000].index(config['n']) if config['n'] in [1000,5000,10000,25000,50000] else 2)
            seed=st.number_input('Graine',0,2147483647,int(config['seed']))
            run=st.form_submit_button('Calculer la simulation',icon=':material/play_arrow:')
        st.caption('Pluie lognormale commune au bassin ; C bêta indépendants entre surfaces et indépendants de la pluie. C = 0 ou 1 reste fixe. CV = écart-type / moyenne. Géométrie et rugosité fixes.')
        if run:
            config.update(cv_i=cvi/100,cv_c=cvc/100,n=int(n),seed=int(seed));commit_config(config)
            with st.spinner('Simulation en cours…'):sim=simulate(config)
            st.session_state.simulation_result=(json.dumps(config,sort_keys=True,ensure_ascii=False),sim);st.rerun()
        if sim:
            risk=sim['system'];st.metric('Au moins un tronçon dépassé',f'{risk["p"]:.2%}')
            st.caption(f'{risk["count"]} / {risk["total"]} ; intervalle exact à 95 % [{risk["low"]:.2%} ; {risk["high"]:.2%}]. Probabilité conditionnelle, pas une fréquence annuelle.')
            if not risk['count']:st.info('Aucun dépassement observé ne signifie pas une probabilité théorique nulle.')
            frame=sim['frame'];q=frame.debit_exutoire_m3s.to_numpy()
            fig=go.Figure(go.Histogram(x=q,nbinsx=45,name='Q simulé'));fig.add_vline(x=sim['capacity'],line_color='#cd593c');fig.update_layout(xaxis_title='Débit exutoire (m³/s)',yaxis_title='Nombre de tirages');chart(fig,'hist')
            sorted_q=np.sort(q);fig=go.Figure(go.Scatter(x=sorted_q,y=(len(q)-np.arange(len(q)))/len(q),name='Survie empirique'));fig.update_layout(xaxis_title='Débit (m³/s)',yaxis_title='Proportion Q ≥ seuil');chart(fig,'survival')
            k=np.unique(np.linspace(100,len(q),60,dtype=int));mask=frame.depassement_reseau.to_numpy();fig=go.Figure(go.Scatter(x=k,y=np.cumsum(mask)[k-1]/k));fig.update_layout(xaxis_title='Tirages cumulés',yaxis_title='Probabilité cumulée réseau');chart(fig,'convergence')
            if sim['sensitivity']:
                labels,values=zip(*sim['sensitivity']);chart(go.Figure(go.Bar(x=values,y=labels,orientation='h')).update_layout(xaxis_title='Corrélation de Spearman',xaxis_range=[-1,1]),'sensitivity')
            else:st.info('Entrées ou résultats fixes : aucune corrélation estimable.')
            st.caption('Sensibilité exploratoire du ratio maximal : ni causalité, ni pourcentage de contribution.')
            st.dataframe(frame[['debit_exutoire_m3s','ratio_max_reseau']].quantile([.05,.5,.95]).rename(index={.05:'P05',.5:'Médiane',.95:'P95'}),width='stretch')
            st.dataframe(pd.DataFrame(sim['risks']).T, width='stretch')
        else:st.info('Aucun résultat actuel. Calculez la simulation après avoir validé les paramètres.')
    elif view=='Comparaison' and ready:
        with st.form('scenario'):
            diameter=st.number_input('Facteur sur tous les diamètres',1.,2.,1.2,.05)
            slope=st.number_input('Facteur sur toutes les pentes',.5,2.,1.,.1)
            reduction=st.slider('Réduction relative des coefficients C (%)',0,80,15)
            compare=st.form_submit_button('Comparer les scénarios',icon=':material/compare_arrows:')
        st.caption('Scénario exploratoire : modifications uniformes, sans coût ni faisabilité technique évalués. Mêmes graines pour la pluie et les quantiles bêta.')
        if compare:
            alternative=copy.deepcopy(config)
            for row in alternative['pipes']:row['diametre_mm']*=diameter;row['pente_pct']*=slope
            for row in alternative['basins']:row['coefficient']*=1-reduction/100
            validate(alternative)
            with st.spinner('Comparaison en cours…'):a=simulate(config);z=simulate(alternative)
            st.session_state.comparison_export=(fingerprint,alternative,a,z)
        comp=st.session_state.get('comparison_export')
        if comp and comp[0]==fingerprint:
            _,alt,a,z=comp
            table=pd.DataFrame({'Scénario':['Actuel','Amélioré'],'P dépassement réseau (%)':[a['system']['p']*100,z['system']['p']*100],
                                'Capacité exutoire (m³/s)':[a['capacity'],z['capacity']],
                                'Q exutoire P95 (m³/s)':[a['frame'].debit_exutoire_m3s.quantile(.95),z['frame'].debit_exutoire_m3s.quantile(.95)]})
            st.dataframe(table,hide_index=True,width='stretch');chart(go.Figure(go.Bar(x=table.Scénario,y=table['P dépassement réseau (%)'],marker_color=['#cd593c','#087aac'])),'compare')
            st.download_button('Exporter la comparaison Excel',excel_bytes({'comparaison':table,'bassins_actuels':b,'reseau_actuel':r,'bassins_scenario':pd.DataFrame(alt['basins']),'reseau_scenario':results(alt)[1], 'parametres':pd.DataFrame([{k:v for k,v in config.items() if k not in ('basins','pipes')},{k:v for k,v in alt.items() if k not in ('basins','pipes')}])}),file_name='hydrorisk-comparaison.xlsx',icon=':material/download:')
    elif view=='Données et exports':
        template=example()
        st.download_button('Modèle Excel fictif',excel_bytes({'bassins':pd.DataFrame(template['basins']),'reseau':pd.DataFrame(template['pipes'])}),file_name='hydrorisk-modele.xlsx',icon=':material/download:')
        st.download_button('Modèle CSV bassins',csv_bytes(pd.DataFrame(template['basins'])),file_name='bassins.csv')
        st.download_button('Modèle CSV réseau',csv_bytes(pd.DataFrame(template['pipes'])),file_name='reseau.csv')
        st.caption('Excel : feuilles bassins et reseau. CSV UTF-8 : séparateur virgule, décimales avec point. Import tabulaire : les paramètres pluie/simulation actuels sont conservés.')
        xlsx=st.file_uploader('Projet Excel',type=['xlsx'],max_upload_size=3)
        bc=st.file_uploader('Bassins CSV',type=['csv'],max_upload_size=1);pc=st.file_uploader('Réseau CSV',type=['csv'],max_upload_size=1)
        js=st.file_uploader('Projet JSON complet',type=['json'],max_upload_size=1)
        confirm=st.checkbox('Confirmer le remplacement des données actuelles')
        if st.button('Importer et valider',icon=':material/upload:',disabled=not confirm):
            draft=copy.deepcopy(config)
            if js:
                draft=json.loads(js.getvalue());validate(draft)
            elif xlsx:
                sheets=pd.read_excel(xlsx,sheet_name=None)
                if not {'bassins','reseau'}<=sheets.keys():raise ValueError('Feuilles bassins et reseau requises.')
                draft.update(basins=clean(sheets['bassins'],BASIN_COLS).to_dict('records'),pipes=clean(sheets['reseau'],PIPE_COLS).to_dict('records'))
            elif bc and pc:
                draft.update(basins=clean(pd.read_csv(bc),BASIN_COLS).to_dict('records'),pipes=clean(pd.read_csv(pc),PIPE_COLS).to_dict('records'))
            else:raise ValueError('Fournissez un JSON, un Excel, ou les deux CSV.')
            draft['label']='Données importées (origine à documenter)';commit_config(draft);st.rerun()
        if ready:
            st.download_button('Projet JSON complet',json.dumps(config,ensure_ascii=False,indent=2),file_name='hydrorisk-projet.json',mime='application/json')
            st.download_button('Résultats réseau CSV',csv_bytes(r),file_name='hydrorisk-reseau.csv')
            sheets={'bassins':b,'reseau':r,'parametres':pd.DataFrame([{k:v for k,v in config.items() if k not in ('basins','pipes')}])}
            if sim:sheets['simulation']=sim['frame']
            st.download_button('Étude Excel',excel_bytes(sheets),file_name='hydrorisk-etude.xlsx')
            st.download_button('Rapport PDF actuel',pdf_bytes(config,sim),file_name='hydrorisk-rapport.pdf',mime='application/pdf')
            if sim:st.download_button('Tirages CSV',csv_bytes(sim['frame']),file_name='hydrorisk-tirages.csv')
            st.caption('Les exports sont produits à partir de la configuration validée. Sans simulation actuelle, le rapport reste déterministe. Le projet et les résultats sont en mémoire de session : téléchargez-les pour les conserver.')
    elif view=='Méthode et tests':
        st.markdown('**Méthode rationnelle** : Q = C × i × A / 360, Q en m³/s, i en mm/h, A en ha. C pondéré = somme(C × A) / somme(A). Pluie uniforme et durée ≥ tc global saisi.')
        st.markdown('**Manning** : Qcap = S × Rh^(2/3) × pente^(1/2) / n. S = πD²/4, Rh = D/4, D en m, pente en m/m (la saisie en % est divisée par 100). Capacité de référence à section pleine ; pas une résolution des pressions ou remous.')
        st.warning(LIMITS)
        st.markdown('**Monte-Carlo** : pluie lognormale de moyenne saisie ; bêta de moyenne C et variance (C × CV)². Variance strictement inférieure à C(1−C). Intervalle binomial exact de Clopper-Pearson à 95 %. Les quantiles des débits ne sont pas cet intervalle.')
        tests=pd.DataFrame([['Rationnelle', 'C = 0,6 ; i = 100 ; A = 2','0,333333 m³/s'],['Linéarité','Doubler i ou A','Doubler Q'],['Cumul','Deux branches 0,10 et 0,20','0,30 m³/s en aval'],['Manning','D = 500 mm ; pente = 1 % ; n = 0,013',f'{float(manning(500,1,.013)):.6f} m³/s'],['Sans incertitude','CV pluie = CV C = 0','Débits constants ; 0 ou 100 % selon le seuil'],['Reproductibilité','Même projet, n et graine','Mêmes tirages']],columns=['Test','Saisie','Attendu'])
        st.dataframe(tests,hide_index=True,width='stretch')
        for name,url in SOURCES.items():st.markdown(f'[{name}]({url})')
    elif ready:
        row=r.set_index('id').loc[outlet]
        values=[('Q exutoire (m³/s)',f'{row.debit_m3s:.3f}'),('Capacité (m³/s)',f'{row.capacite_m3s:.3f}'),('Utilisation exutoire',f'{row.utilisation_pct:.1f} %'),('P dépassement réseau',f'{sim["system"]["p"]:.1%}' if sim else 'À calculer')]
        for idx,(col,(label,value)) in enumerate(zip(st.columns(4),values)):
            with col,st.container(key=f'kpi_{idx}'):st.metric(label,value)
        for col,(label,dest,icon) in zip(st.columns(3),[('Étudier la pluie','Pluie et ruissellement','rainy'),('Lancer une simulation','Simulation','query_stats'),('Comparer','Comparaison','compare_arrows')]):
            with col:st.button(label,icon=f':material/{icon}:',on_click=navigate,args=(dest,),width='stretch')
        chart(network_chart(config),'network_dashboard')
        fig=go.Figure();fig.add_bar(x=r.id,y=r.debit_m3s,name='Débit');fig.add_bar(x=r.id,y=r.capacite_m3s,name='Capacité');fig.update_layout(yaxis_title='m³/s',barmode='group');chart(fig,'capacity')
        limiting=r.sort_values('utilisation_pct',ascending=False).iloc[0]
        st.info(f'Tronçon le plus sollicité : {limiting.id} ({limiting.utilisation_pct:.1f} %). Les débits sont des pointes théoriques simultanées, sans stockage.')


if __name__=='__main__':
    try:main()
    except (ValueError,KeyError,TypeError,OverflowError, pd.errors.ParserError,UnicodeDecodeError) as exc:
        st.error(f'Données non appliquées : {exc}')
    except Exception:
        st.error('Opération impossible. Vérifiez le format des fichiers et les données ; les autres vues restent accessibles.')
