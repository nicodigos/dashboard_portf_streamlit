import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from streamlit_plotly_events import plotly_events
import pandas as pd

def select_reference(df, selected_points):
    try:
        inner_df = df[df.year == (selected_points[0]['x'] - 1)]
        if inner_df.shape[0] != 0:
            return float(inner_df[cuantitative_variable].sum())
        else:
            return sum(map(lambda x: x['y'], selected_points))
    except IndexError:
        return 0
    
def selected_points_year(selected_points):
    try:
        return selected_points[0]['x']
    except IndexError:
        return 0
    


st.set_page_config(
    layout="wide",                      
)

dataframes = {'Labour Force vs Total Population': 'labour_force_vs_total_population.csv',
              'Part time vs Full time': 'parttime_vs_fulltime.csv',
              'Employment vs Unemployment': 'employment_vs_unemployment.csv'     }

selected_df = st.selectbox('Select Dataframe', dataframes.keys(), index=2)

df = pd.read_csv(f"data/{dataframes[selected_df]}")
cualitative_variable = 'variable'
cuantitative_variable = 'value'
base_df = pd.read_csv(f"data/{dataframes[selected_df][:-4]}_second.csv")

provinces = st.multiselect(
        label="Select provinces for display",
        options=df['GEO'].unique().tolist(),
        default=df['GEO'].unique().tolist(),
        label_visibility = "hidden"
    )


col1, col2, col3 = st.columns([0.42, 0.42, 0.16], vertical_alignment='center', gap='medium')
with col1: 
    slider = st.slider("Year", 
              min_value=df.year.min(), 
              max_value=df.year.max(),
              value=(df.year.min(), df.year.max()), 
              step=1)

with col2:
    option = st.multiselect(
    "Select categories to show",
    options = df[cualitative_variable].unique().tolist(),
    default = df[cualitative_variable].unique().tolist(),
)



# def filter_dataframe(df, option, column):
df_modif = df[df[cualitative_variable].isin(option)]
df_modif = df_modif[(df_modif.year >= slider[0]) & (df_modif.year <= slider[1])]
df_modif = df_modif[df['GEO'].isin(provinces)]
df_modif = df_modif[['year', 'variable', 'value']].groupby(['year', 'variable']).sum().reset_index()

# base_df_modif = base_df[base_df[cualitative_variable].isin(option)]
# base_df_modif = base_df_modif[base_df['GEO'].isin(provinces)]
# base_df_modif = base_df_modif[['year', 'variable', 'value']].groupby(['year', 'variable']).sum().reset_index()


# def filter_changes(df):
fig = px.area(df_modif, x="year", 
            y=cuantitative_variable, 
            color=cualitative_variable, 
            template="plotly_dark",
            category_orders={cualitative_variable: option}
            )
fig.update_traces(mode="lines", hovertemplate='%{y:,.0f}', stackgroup='one')
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
    # margin=dict(l=0, r=0, t=0, b=0),
    # colorway=set2_palette,
    xaxis=dict(
        type='linear',
        tickmode='linear',  # Automatically show integer ticks
        dtick=1,  # Specify step for integer ticks
    ),# Configuración de hover sin interferencia

    )



selected_points = plotly_events(
    fig, 
    hover_event=True,       # Capturar eventos de hover
)


with col3:
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

        margin=dict(l=0, r=0, t=30, b=0), # Set all margins to 0

        width=100,  # Ancho en píxeles
        height=120
    )

    st.plotly_chart(total, use_container_width=False)




# st.write("Datos capturados en el hover:", selected_points)



second_col1, second_col2 = st.columns(2)

with second_col1:
    try:
        pie = px.pie(values=list(map(lambda x: x['y'], selected_points)), 
                    names=option,
                    template="plotly_dark",
                    title=f"Share, {selected_points_year(selected_points)}")
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
            reference = reference[[cualitative_variable, cuantitative_variable]]\
                                .groupby(cualitative_variable)\
                                .sum()\
                                .to_dict()[cuantitative_variable]
            results_dict = {}
       
            for k, v in bars_dict.items():
                results_dict[k] = ((bars_dict[k] - reference[k])/reference[k])


            
            result = {
                'category': list(results_dict.keys()),
                'growth': list(results_dict.values())
            }

            bars = px.bar(result,
                          x='category', 
                          y='growth',
                          title=f"Growth, {selected_points_year(selected_points)}")

            bars.update_traces(

                    hovertemplate=[f"{value:.2%}" for value in result['growth']],
                    text=[f"{value:.2%}" for value in result['growth']],
                    marker_color=['#ff6961' if value < 0 else '#8edcb9' for value in result['growth']]
                        )

            st.plotly_chart(bars)
    except Exception:
        pass


# base_df_modif = base_df_modif[base_df_modif.year.str[:4] == str(selected_points_year(selected_points))]

# month=px.area(base_df_modif, x="year", 
#             y=cuantitative_variable, 
#             color=cualitative_variable, 
#              template="plotly_dark",
#             category_orders={cualitative_variable: option})

# month.update_traces(mode="lines", hovertemplate='%{y:,.0f}', stackgroup='one')
# month.update_layout(hovermode="x")
# month.update_layout(
#     paper_bgcolor="rgba(0,0,0,0)",  # Fondo del lienzo transparente
#     # plot_bgcolor="rgba(0,0,0,0)"    # Fondo del área del gráfico transparente
# )
# month.update_layout(legend=dict(
#     orientation="h",
#     yanchor="bottom", 
#     y=1.02,
#     xanchor="right",
#     x=1
    # ),
    # margin=dict(l=0, r=0, t=0, b=0),
    # colorway=set2_palette,
    # xaxis=dict(
    #     type='linear',
    #     tickmode='linear',  # Automatically show integer ticks
    #     dtick=1,  # Specify step for integer ticks
    # ),# Configuración de hover sin interferencia

    # )


# st.plotly_chart(month)