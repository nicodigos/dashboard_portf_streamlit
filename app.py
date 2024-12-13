import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import pandas as pd

def select_reference(df, selected_points):
    try:
        inner_df = df[df.year == (selected_points[0]['x'] - 1)]
        if inner_df.shape[0] != 0:
            return float(inner_df['Total_Population'].sum())
        else:
            return sum(map(lambda x: x['y'], selected_points))
    except IndexError:
        return 0
    
def selected_points_year(selected_points):
    try:
        return selected_points[0]['x'] - 1
    except IndexError:
        return 0
    
set2_palette = px.colors.qualitative.Set2

df = pd.read_csv("data/data_unemployment_first_chart.csv")

st.set_page_config(
    layout="wide",                      
)

option = st.multiselect(
    "How would you like to be contacted?",
    options = df.GEO.unique().tolist(),
    placeholder="Select contact method...",
    default = df.GEO.unique().tolist(),
)

col1, col2 = st.columns([0.84, 0.16], vertical_alignment='center', gap='medium')
with col1: 
    slider = st.slider("Year", 
              min_value=df.year.min(), 
              max_value=df.year.max(),
              value=(df.year.min(), df.year.max()), 
              step=1)




# def filter_dataframe(df, option, column):
df_modif = df[df['GEO'].isin(option)]
df_modif = df_modif[(df_modif.year >= slider[0]) & (df_modif.year <= slider[1])]


# def filter_changes(df):
fig = px.area(df_modif, x="year", 
            y="Total_Population", 
            color="GEO", 
            template="plotly_dark",
            )
fig.update_traces(mode="lines", hovertemplate='%{y:,.0f}')
fig.update_layout(hovermode="x")
fig.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",  # Fondo del lienzo transparente
    # plot_bgcolor="rgba(0,0,0,0)"    # Fondo del área del gráfico transparente
)
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
    ),
    margin=dict(l=0, r=0, t=0, b=0),
    # colorway=set2_palette,
    xaxis=dict(
        type='linear',
        tickmode='linear',  # Automatically show integer ticks
        dtick=1,  # Specify step for integer ticks
    ))




selected_points = plotly_events(
    fig, 
    hover_event=True,       # Capturar eventos de hover
)


with col2:
    total = go.Figure(go.Indicator(
    mode = "number+delta",
    value = sum(map(lambda x: x['y'], selected_points)),
    number = { 'valueformat': ',.0f' },
    delta = {'relative': True, 
             'position': "top", 
             "valueformat": ".2%",
             'reference': select_reference(df_modif, selected_points)},
    title = f"Total {selected_points_year(selected_points)}",
    domain = {'x': [0, 1], 'y': [0, 1]}))

    total.update_layout(

        margin=dict(l=0, r=0, t=0, b=0), # Set all margins to 0
        width=100,  # Ancho en píxeles
        height=120
    )

    st.plotly_chart(total, use_container_width=False)




# st.write("Datos capturados en el hover:", selected_points)



second_col1, second_col2 = st.columns(2, border=True)

with second_col1:
    try:
        pie = px.pie(values=list(map(lambda x: x['y'], selected_points)), 
                    names=option,
                    template="plotly_dark",)
        pie.update_traces(
            sort=False,
        )

        st.plotly_chart(pie)
    except ValueError:
        pass

with second_col2:
    try:
        if selected_points[0]['x'] == slider[0]:
            pass
        else:
            bars_dict = dict(zip(option, list(map(lambda x: x['y'], selected_points))))
            # bars_df = pd.DataFrame(bars_dict)
            reference = df_modif[df_modif.year == selected_points[0]['x']-1]
            reference = reference[['GEO', 'Total_Population']]\
                                .groupby('GEO')\
                                .sum()\
                                .to_dict()['Total_Population']
            results_dict = {}
       
            for k, v in bars_dict.items():
                results_dict[k] = ((bars_dict[k] - reference[k])/reference[k])


            
            result = {
                'category': list(results_dict.keys()),
                'growth': list(results_dict.values())
            }

            bars = px.bar(result,
                          y='category', 
                          x='growth',)

            bars.update_traces(
                    hovertemplate=[f"{value:.2%}" for value in result['growth']],
                    text=[f"{value:.2%}" for value in result['growth']],
                    marker_color=['#ff0000' if value < 0 else '#00ff00' for value in result['growth']]
                        )

            st.plotly_chart(bars)
    except Exception:
        pass