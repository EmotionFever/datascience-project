from pandas import read_csv, Timestamp
from numpy import nan
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import subplots, savefig, show
from ds_charts import get_variable_types, HEIGHT
from matplotlib.pyplot import figure, savefig, show, title
from seaborn import heatmap

register_matplotlib_converters()
#filename = "data/set1_NYC_collisions_tabular.csv"
filename = 'lab01_data_profiling\data\set1_NYC_collisions_tabular.csv'
data = read_csv(filename, parse_dates=['CRASH_DATE'], infer_datetime_format=True)

data['CRASH_MONTH'] = data['CRASH_DATE'].dt.month
data['CRASH_DAY'] = data['CRASH_DATE'].dt.day
data.drop(columns=['CRASH_DATE'], inplace=True)
data['CRASH_TIME'] = data['CRASH_TIME'].str.split(':').str[0].astype(int)
data['PERSON_INJURY'] = data['PERSON_INJURY'].astype('category').cat.codes

complaint_encode = { # how severe are the complaints
    'Complaint of Pain or Nausea': 1,
    'Minor Bleeding':2,
    'None Visible': 0,
    'Contusion - Bruise': 2,
    'Severe Bleeding': 7,
    'Internal': 6,
    'Severe Lacerations':7,
    'Abrasion': 2,
    'Fracture - Distorted - Dislocation': 4,
    'Whiplash': 2, #passes after some weeks
    'Unknown': nan,
    'Concussion': 10,
    'Crush Injuries': 6,
    'Minor Burn': 2,
    'Paralysis': 9,
    'Amputation': 7,
    'Moderate Burn': 4,
    'Severe Burn': 6,
    'Does Not Apply': nan
}
data["COMPLAINT"].replace(complaint_encode, inplace=True)

emotion_status_encode = { # how well is the person emotionally
    'Conscious':10,
    'Apparent Death': 0,
    'Semiconscious': 4,
    'Shock': 8,
    'Unknown': nan,
    'Unconscious': 2,
    'Incoherent': 7,
    'Does Not Apply': nan
}
data["EMOTIONAL_STATUS"].replace(emotion_status_encode, inplace=True)

safety_equipment_encode = { # how secure is the equipement
    'Lap Belt & Harness': 10,
    'Helmet (Motorcycle Only)': 1,
    'Air Bag Deployed/Lap Belt/Harness':10,
    'Unknown':nan,
    'None': 0,
    'Lap Belt':10,
    'Helmet Only (In-Line Skater/Bicyclist)': 1,
    'Child Restraint Only': 7,
    'Helmet/Other (In-Line Skater/Bicyclist)':5,
    'Air Bag Deployed/Child Restraint':10,
    'Air Bag Deployed': 3,
    'Other': 5,
    'Harness': 10,
    'Air Bag Deployed/Lap Belt': 10,
    'Pads Only (In-Line Skater/Bicyclist)': 4,
    'Stoppers Only (In-Line Skater/Bicyclist)': 1
}
#data["SAFETY_EQUIPMENT"].replace(safety_equipment_encode, inplace=True)
data.to_csv('lab02_data_preparation\ew_data\set1_symbolic_to_numeric.csv', index=False)
# lab02_data_preparation\ew_data

