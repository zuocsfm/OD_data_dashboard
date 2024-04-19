import time  # to simulate a real time data, time loop

import streamlit as st
import pandas as pd
import pyproj
import numpy as np
import plotly.express as px
from streamlit_extras.metric_cards import style_metric_cards
import pydeck as pdk
import math
import warnings
warnings.filterwarnings('ignore')


import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Transport Data Analysis Dashboard",
    page_icon=":bar_chart:",
    layout="wide",
)


st.title("Trip Data Analysis")

data = pd.read_csv("./data/origin-destination.csv", sep=';')
transport_mode_list = data['mode'].unique().tolist()
departure_time_list = data['departure_time'].unique().tolist()
travel_time_list = data['travel_time'].unique().tolist()

# ------------------------------------------------------------------------
#  calculate the coordinates
# ------------------------------------------------------------------------

# convert the coordinates to latitude and longitude
proj = pyproj.Transformer.from_crs( 2154, 4326, always_xy=True)

# get the latitude and longitude
data['origin'] = data.apply(lambda row: proj.transform(row['origin_x'], row['origin_y']), axis=1)
data[['origin_lon', 'origin_lat']] = pd.DataFrame(data['origin'].tolist(), index=data.index)

# get the latitude and longitude
data['destination'] = data.apply(lambda row: proj.transform(row['destination_x'], row['destination_y']), axis=1)
data[['destination_lon', 'destination_lat']] = pd.DataFrame(data['destination'].tolist(), index=data.index)


# ------------------------------------------------------------------------
#  sidebar
# ------------------------------------------------------------------------

with st.sidebar:
    # st.write("Data source: MATSim (https://www.matsim.org/)")
    st.title("Data filters")

    new_departure = st.slider(label="Select a departure time range (hour):",
                           min_value=6,
                           max_value=24,
                           value=(6, 20))

    new_departure = tuple((i*60*60) for i in new_departure)

    new_travel_time = st.slider(label='Select a travel duration time range (hour):',
                                   min_value=0,
                                   max_value=22,
                                   value=(0,22))
    new_travel_time = tuple((i*60*60) for i in new_travel_time)

    new_distance = st.slider(label='Select a routed distance range (km):',
                                   min_value=0,
                                   max_value=172,
                                   value=(0,172))
    new_distance = tuple((i * 1000) for i in new_distance)

    new_mode = st.multiselect("Choose transport mode:", transport_mode_list, transport_mode_list )

# filter data according to user selection
selected_subset = (data['departure_time'].between(*new_departure)) \
                  & (data['travel_time'].between(*new_travel_time)) & (data['mode'].isin(new_mode)\
                    & (data['routed_distance'].between(*new_distance)))

selected_subset = data[selected_subset]



# ------------------------------------------------------------------------
#  Summary - row 1
# ------------------------------------------------------------------------

row1_1, row1_2, row1_3, row1_4, row1_5, row1_6 = st.columns(6)

# display the statistics

trip_number = len((selected_subset['person_id'].astype(str) + "_" + selected_subset['person_trip_id'].astype(str)).unique())
row1_1.metric("Number of trips", trip_number)

leg_number = len(selected_subset.index)
row1_2.metric("Number of legs", leg_number)

person_number = len(selected_subset['person_id'].unique())
row1_3.metric("Number of agents", person_number)

travel_time_average = selected_subset['travel_time'].mean()
row1_4.metric("Average travel time (minutes)", (travel_time_average/60).round(2))

routed_distance_ave = selected_subset['routed_distance'].mean().round(2)
row1_5.metric("Average routed distance (km)", (routed_distance_ave/1000).round(2))

row1_6.metric("Average speed (km/h)", (routed_distance_ave/travel_time_average*60).round(2))

style_metric_cards()

# ------------------------------------------------------------------------
#  Chart - row 2
# ------------------------------------------------------------------------

row2_1, row2_2, row2_3 = st.columns(3)

# ------------------------------------------------------------------------
#  Chart
# ------------------------------------------------------------------------

# chart1, chart2, chart3, chart4 = st.columns(4)
# travel_mode = selected_subset.groupby(['mode'])['mode'].count()
# travel_mode = pd.DataFrame({'mode':travel_mode.index, 'number':travel_mode.values})

# row2_3.write("Number of Trips by Travel Mode")
# row2_3.bar_chart(travel_mode, x='mode', y='number')



# ------------------------------------------------------------------------
#  Map
# ------------------------------------------------------------------------

# draw the origins
# get the latitude and longitude
selected_subset['origin'] = selected_subset.apply(lambda row: proj.transform(row['origin_x'], row['origin_y']), axis=1)
selected_subset[['origin_lon', 'origin_lat']] = pd.DataFrame(selected_subset['origin'].tolist(), index=selected_subset.index)

# chart1.write("The location of origins")
# chart1.map(selected_subset, latitude='origin_lat', longitude='origin_lon', size = 10, color='#00445f')

# draw the destinations
# get the latitude and longitude
selected_subset['destination'] = selected_subset.apply(lambda row: proj.transform(row['destination_x'], row['destination_y']), axis=1)
selected_subset[['destination_lon', 'destination_lat']] = pd.DataFrame(selected_subset['destination'].tolist(), index=selected_subset.index)

# chart2.write("The location of destinations")
# chart2.map(selected_subset, latitude='destination_lat', longitude='destination_lon')

# ------------------------------------------------------------------------
#  Flow map
# ------------------------------------------------------------------------

# draw the flow map
GREEN_RGB = [98, 115, 19, 80]
RED_RGB = [183, 53, 45, 80]

row2_1.subheader("The origins and destinations")
row2_1.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        latitude=42,
        longitude=9.1,
        zoom=8,
        pitch=170,
    ),
    layers=[
        pdk.Layer(
           "ArcLayer",
            data=selected_subset,
            get_width="S000 * 2",
            get_source_position=["origin_lon", "origin_lat"],
            get_target_position=["destination_lon", "destination_lat"],
            get_tilt=15,
            get_source_color=RED_RGB,
            get_target_color=GREEN_RGB,
            pickable=True,
            auto_highlight=True,
        ),
    ],
    tooltip={
        'html': '<b>Person id:</b> {person_id}<br><b>Trip id:</b> {person_trip_id}<br><b>Leg index:</b> {leg_index}',
        'style': {
            'color': 'white'
        }
    }
))

# ------------------------------------------------------------------------
#  map the stops
# ------------------------------------------------------------------------
#

# calculate breaks

selected_subset['arrival_time'] = selected_subset['departure_time'] + selected_subset['travel_time']
max_legs = max(selected_subset['leg_index'].unique().tolist())

df_transitional_stops = pd.DataFrame(columns=['person_id', 'stop_index', 'lat', 'lon', 'end_time', 'start_time', 'duration'])
df_activity_stops = pd.DataFrame(columns=['person_id', 'stop_index', 'lat', 'lon', 'end_time', 'start_time', 'duration'])

for i in range(1, max_legs+1):
    # find arrival and departure trip pairs
    df_arrival = selected_subset.loc[selected_subset['leg_index'] == i]
    df_departure = selected_subset.loc[selected_subset['leg_index'] == (i-1)]

    arrival_person = df_arrival['person_id'].unique().tolist()
    departure_person = df_departure['person_id'].unique().tolist()

    # calculate the person made stops
    common_person = set(arrival_person) & set(departure_person)

    for p in common_person:
        stop_end = selected_subset[(selected_subset['person_id'] == p) & (selected_subset['leg_index'] == i)]
        stop_start = selected_subset[(selected_subset['person_id'] == p) & (selected_subset['leg_index'] == (i - 1))]

        stop_lat = stop_end['origin_lat']#.tolist()[0]
        stop_lon = stop_end['origin_lon']#.tolist()[0]
        stop_end_time = int(stop_end['departure_time'].tolist()[0])
        stop_start_time = int(stop_start['arrival_time'].tolist()[0])
        stop_duration = stop_end_time - stop_start_time
        # stop_mode = stop_end['mode'] # a stop does not have a mode, this is only used to generate a dataframe correctly

        # new_stop = {}



        # the different trip_id indicates an agent had activity
        if stop_end['person_trip_id'].tolist()[0] == stop_start['person_trip_id'].tolist()[0]:
            new_transitional_stop = {}
            new_transitional_stop['person_id'] = p
            new_transitional_stop['stop_index'] = i
            new_transitional_stop['lat'] = stop_lat
            new_transitional_stop['lon'] = stop_lon
            new_transitional_stop['end_time'] = stop_end_time
            new_transitional_stop['start_time'] = stop_start_time
            new_transitional_stop['duration'] = stop_duration
            # new_transitional_stop['mode'] = stop_mode

            df_new_transitional_stop = pd.DataFrame.from_dict(new_transitional_stop)  #
            df_transitional_stops = pd.concat([df_transitional_stops, df_new_transitional_stop], ignore_index=True)
        # the identical trip id indicates an agent had activity
        else:
            new_activity_stop = {}
            new_activity_stop['person_id'] = p
            new_activity_stop['stop_index'] = i
            new_activity_stop['lat'] = stop_lat
            new_activity_stop['lon'] = stop_lon
            new_activity_stop['end_time'] = stop_end_time
            new_activity_stop['start_time'] = stop_start_time
            new_activity_stop['duration'] = stop_duration
            # new_activity_stop['mode'] = stop_mode

            df_new_activity_stop = pd.DataFrame.from_dict(new_activity_stop)  #
            df_activity_stops = pd.concat([df_activity_stops, df_new_activity_stop], ignore_index=True)

# TODO: show the stop duration in the visualization instead of number of stops
row2_3.subheader("The transitional stops")
row2_3.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        latitude=41.9,
        longitude=9.1,
        zoom=8,
        pitch=170,
    ),
    layers=[
        pdk.Layer(
           'HexagonLayer',
           data=df_transitional_stops,
           get_position='[lon, lat]',
           radius=500,
           elevation_scale=4,
           elevation_range=[0, 5000],
           pickable=True,
           extruded=True,
        ),
    ],
    tooltip={
            'html': '<b>Number of stops:</b> {elevationValue}',
            'style': {
                'color': 'white'
            }
        }
))

# TODO: show the stop duration in the visualization instead of number of stops
row2_2.subheader("The activity stops")
row2_2.pydeck_chart(pdk.Deck(
    map_style=None,
    initial_view_state=pdk.ViewState(
        latitude=41.9,
        longitude=9.1,
        zoom=8,
        pitch=170,
    ),
    layers=[
        pdk.Layer(
           'HexagonLayer',
           data=df_activity_stops,
           get_position='[lon, lat]',
           radius=500,
           elevation_scale=4,
           elevation_range=[0, 5000],
           pickable=True,
           extruded=True,
        ),
    ],
    tooltip={
            'html': '<b>Number of stops:</b> {elevationValue}',
            'style': {
                'color': 'white'
            }
        }
))

# ------------------------------------------------------------------------
#  Show travel mode per hour (while trips)
# ------------------------------------------------------------------------
st.subheader("Travel modes in every hour")
# initialize the traval mode per hour matrix
travel_mode_hour = pd.DataFrame(0, index=np.arange(25), columns = ["car", "walk", "pt","car_passenger", "bike"])

# calculate travel mode per hour
def calc_mode_hour(mode, start_time, duration, matrix):
    end_time = start_time + duration
    start_hour = math.floor(start_time/60/60)
    end_hour = math.ceil(end_time/60/60)
    matrix.loc[start_hour:(end_hour + 1), mode] = matrix.loc[start_hour:(end_hour + 1), mode] + 1
    return matrix

for index, row in selected_subset.iterrows():
    travel_mode_hour = calc_mode_hour(row['mode'], row['departure_time'], row['travel_time'],travel_mode_hour)

st.bar_chart(travel_mode_hour, color=['#7fc97f','#beaed4','#fdc086','#ffff99','#386cb0'])


# ------------------------------------------------------------------------
#  Show the Raw data
# ------------------------------------------------------------------------
# delete the intermediate columns
selected_subset.drop(['origin','destination', 'origin_lon', 'origin_lat', 'destination_lat', 'destination_lon', 'arrival_time'], axis='columns', inplace=True)

st.subheader("Filtered dataset")

st.dataframe(selected_subset, width=2000)


# ------------------------------------------------------------------------
#  data download
# ------------------------------------------------------------------------

@st.cache_data
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(sep=';').encode('utf-8')

csv = convert_df(selected_subset)

with st.sidebar:
    st.write("\n")
    st.download_button(
        label="Download filtered data as CSV",
        data=csv,
        file_name='filtered_eqasim_data.csv',
        mime='text/csv',
    )







