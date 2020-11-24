import streamlit as st
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import networkx as nx
import numpy as np
import requests 
st.header("Du Chemin Lost Voices Cadence Data")

# st.cache speeds things up by holding data in cache
#@st.cache
def get_data():
    url = "https://raw.githubusercontent.com/RichardFreedman/LostVoicesCadenceViewer/main/LV_CadenceData.csv"
    df = pd.read_csv(url)
    cadence_json =  requests.get("https://raw.githubusercontent.com/bmill42/DuChemin/master/phase1/data/duchemin.similarities.json").json()
    df['similarity'] = cadence_json
    return df 
    
df = get_data()

# Dialogue to Show Raw Data as Table

if st.sidebar.checkbox('Show Complete Data Frame'):
	st.subheader('Raw data')
	st.write(df)

#tones = df['cadence_final_tone'].drop_duplicates()
tones = df[["cadence_final_tone", "cadence_kind", "final_cadence", "composition_number"]]


# This displays unfiltered 

all_tone_diagram = alt.Chart(tones).mark_circle().encode(
    x='final_cadence',
    y='composition_number',
    color='cadence_final_tone',
    shape='cadence_kind'
)

if st.sidebar.checkbox('Show All Pieces with Their Cadences'):
	st.subheader('All Pieces with Cadences')
	st.altair_chart(all_tone_diagram, use_container_width=True)


# Dialogue to Select Cadence by Final Tone
st.subheader('Selected Cadences by Final Tone')


# Create a list of possible values and multiselect menu with them in it.

#cadence_list = tones['cadence_final_tone']
cadence_list = tones['cadence_final_tone'].unique()
cadences_selected = st.sidebar.multiselect('Select Tone(s)', cadence_list)

# Mask to filter dataframe
mask_cadences = tones['cadence_final_tone'].isin(cadences_selected)

tone_data = tones[mask_cadences]

# This is for filtered tones (just oned)
tone_diagram = alt.Chart(tone_data).mark_circle().encode(
    x='cadence_kind',
    y='composition_number',
    color='final_cadence',
    #shape='final_cadence',
    tooltip=['cadence_kind', 'composition_number', 'final_cadence']
)

st.altair_chart(tone_diagram, use_container_width=True)


# This displays choice of piece 
st.subheader('Selected Pieces')

piece_list = tones['composition_number'].unique()
pieces_selected = st.sidebar.multiselect('Select Piece(s)', piece_list)

# Mask to filter dataframe
mask_pieces = tones['composition_number'].isin(pieces_selected)

piece_data = tones[mask_pieces]
piece_diagram = alt.Chart(piece_data).mark_circle().encode(
    x='cadence_final_tone',
    y='cadence_kind',
    color='final_cadence',
    #shape='final_cadence'
)

st.altair_chart(piece_diagram, use_container_width=True)


###
#Graph Visualization
###
cadence_graph = nx.Graph()

# Add a node for each cadence 
for index, row in df.iterrows():
    cadence_graph.add_node(row.phrase_number, size=1.5)

# Add all the edges
for index, row in df.iterrows():
    for i in row.similarity:
        cadence_graph.add_edge(row.phrase_number, df['phrase_number'][i], weight=2)

# Get positions for the nodes in G
pos_ = nx.spring_layout(cadence_graph)

def make_edge(x, y, text, width):
    
    '''Creates a scatter trace for the edge between x's and y's with given width

    Parameters
    ----------
    x    : a tuple of the endpoints' x-coordinates in the form, tuple([x0, x1, None])
    
    y    : a tuple of the endpoints' y-coordinates in the form, tuple([y0, y1, None])
    
    width: the width of the line

    Returns
    -------
    An edge trace that goes between x0 and x1 with specified width.
    '''
    return  go.Scattergl(x         = x,
                       y         = y,
                       line      = dict(width = width,
                                   color = 'cornflowerblue'),
                       hoverinfo = 'text',
                       text      = ([text]),
                       mode      = 'lines')   

# For each edge, make an edge_trace, append to list
edge_trace = []
for edge in cadence_graph.edges():
    char_1 = edge[0]
    char_2 = edge[1]

    x0, y0 = pos_[char_1]
    x1, y1 = pos_[char_2]

    text   = char_1 + '--' + char_2 
    
    trace  = make_edge([x0, x1, None], [y0, y1, None], text,
                        0.3*cadence_graph.edges()[edge]['weight']**1.75)

    edge_trace.append(trace)

# Make a node trace
node_trace = go.Scattergl(x = [],
                        y = [],
                        text = [],
                        textposition = "top center",
                        textfont_size = 10,
                        mode      = 'markers+text',
                        hoverinfo = 'none', 
                        marker    = dict(color = [],
                                         size  = [],
                                         line  = None))

# For each node in cadence_graph, get the position and size and add to the node_trace
for node in cadence_graph.nodes():
    x, y = pos_[node]
    node_trace['x'] += tuple([x])
    node_trace['y'] += tuple([y])
    node_trace['marker']['color'] += tuple(['cornflowerblue'])
    node_trace['marker']['size'] += tuple([5*cadence_graph.nodes()[node]['size']])
    # node_trace['phrase_number'] += tuple(['<b>' + node + '</b>'])

layout = go.Layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)'
)


fig = go.Figure(layout = layout)

for trace in edge_trace:
    fig.add_trace(trace)

fig.add_trace(node_trace)

fig.update_layout(showlegend = False)

fig.update_xaxes(showticklabels = False)

fig.update_yaxes(showticklabels = False)

st.plotly_chart(fig)