import numpy as np

HUGE_FLOAT = 1.0e20
MISSING_FLOAT = -99.00
MISSING_INT = -99
LARGE_FLOAT = 10000000

# Labels for columns in a dataset as well as the formatters.
DATA_LABELS_EST = []
DATA_LABELS_EST += ["Identifier", "Period", "Choice", "Wage"]
DATA_LABELS_EST += ["Experience_Work", "Years_Schooling"]
DATA_LABELS_EST += ["Lagged_Choice"]

# There is additional information available in a simulated dataset.
DATA_LABELS_SIM = DATA_LABELS_EST[:]
DATA_LABELS_SIM += ["Type"]
DATA_LABELS_SIM += ["Total_Reward_1", "Total_Reward_2"]
DATA_LABELS_SIM += ["Total_Reward_3"]
DATA_LABELS_SIM += ["Systematic_Reward_1", "Systematic_Reward_2"]
DATA_LABELS_SIM += ["Systematic_Reward_3"]
DATA_LABELS_SIM += ["Shock_Reward_1", "Shock_Reward_2"]
DATA_LABELS_SIM += ["Shock_Reward_3"]
DATA_LABELS_SIM += ["Discount_Rate", "General_Reward", "Common_Reward"]
DATA_LABELS_SIM += ["Immediate_Reward_1", "Immediate_Reward_2", "Immediate_Reward_3"]


DATA_FORMATS_EST = dict()
for key_ in DATA_LABELS_EST:
    DATA_FORMATS_EST[key_] = np.int
    if key_ in ["Wage"]:
        DATA_FORMATS_EST[key_] = np.float

DATA_FORMATS_SIM = dict(DATA_FORMATS_EST)
for key_ in DATA_LABELS_SIM:
    if key_ in DATA_FORMATS_SIM.keys():
        continue
    elif key_ in ["Type"]:
        DATA_FORMATS_SIM[key_] = np.int
    else:
        DATA_FORMATS_SIM[key_] = np.float
