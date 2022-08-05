import sys
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold, KFold, ShuffleSplit
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, \
    GradientBoostingClassifier, RandomForestRegressor, VotingClassifier
# from xgboost import XGBClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.linear_model import LogisticRegression, BayesianRidge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline

from sklearn_custom.imputers.SimpleImputer import SimpleImputer
from sklearn_custom.preprocessing.MinMaxScaler import MinMaxScaler
from sklearn_custom.transformers.ColumnTransformer import ColumnTransformer
from sklearn_custom.encoders.CountFrequencyEncoder import CountFrequencyEncoder
from sklearn_custom.feature_selection.SelectKBest import SelectKBest
from sklearn_custom.feature_selection.SmartCorrelatedSelection import SmartCorrelatedSelection
from sklearn_custom.feature_selection.DropConstantFeatures import DropConstantFeatures
from sklearn.metrics import make_scorer, accuracy_score, f1_score, precision_score, max_error, r2_score
from sklearn.compose import make_column_selector

# in each run this key should be changed. If not the results from the run with this unique_key will be overwritten!
unique_key = "2022-08-05 12:30"

# change path (add prefix) if request comes not from apptainer 'run' method
calling_system = os.getenv('SYS')
if calling_system == "APPTAINER" and os.getenv('RUN') == "TRUE":
    prefix = ""
else:
    prefix = "../"

# *********************************************************************************************************************
# ********************************************* LOGGING ***************************************************************
# *********************************************************************************************************************

logging_path = f'{prefix}data/output/'

loging_config = {
    "handlers": [
        {"sink": sys.stdout, "level": "INFO"},
        {"sink": f"{os.path.join(logging_path, 'logfile.log')}",
         "format": "\n{time:YYYY-MM-DD at HH:mm:ss} | level={level} | {file} | {function} at line {line} \n-> {message}",
         "rotation": "1 day", "backtrace": True, "diagnose": True, "level": "INFO"},
    ]
}

# *********************************************************************************************************************
# ***************************************** STATISTICAL_TESTS *********************************************************
# *********************************************************************************************************************

statistical_tests = {
    'input_file': f'{prefix}data/output/preprocessed_data.pkl',
    'output_folder': f'{prefix}data/output/statisticalTests_{unique_key}'
}

# *********************************************************************************************************************
# ******************************************* BUILD_ML_MODELS *********************************************************
# *********************************************************************************************************************

ml_models_path = {
    "input_file": f"{prefix}data/output/preprocessed_data.pkl",
    "output_folder": f"{prefix}data/output/ml_models_{unique_key}"
}

target = 'diagnosis'  # diagnosis_location, diagnosis -> if 'diagnosis_location' scoring function has to be adjusted to multi-output
subset = 'South Africa'  # ('South Africa' , 'Switzerland', 'both')  # both: african kids as train and swiss kids as testset

# scoring = {'f1': make_scorer(f1_score, average='micro'), 'accuracy': 'accuracy'}
scoring = {'accuracy': 'accuracy', 'prec': make_scorer(precision_score)}
refit = 'accuracy'
return_train_score = True

if subset == "South Africa":

    # include = False
    # include_exclude_columns = [
    #     'antibiotic_exposure_courses',  # exclude because does not run with explainerdashboard (reason: unknown)
    #     'location',
    #     'PID',
    #     'RNASeq',
    #     'bsa',
    #     'scorad',
    #     'eczema_age',
    #     'eczema_ever',
    #     'eczema_whodiagnosed',
    #     'medication_antihistamines',
    #     'medication_antihistamines_last',
    #     'medication_steroidcreams',
    #     'parental_income',  # overfit -> parental_income_class
    #     'parental_education_level',  # ?!
    #     'medicalhistory_comorbidities',  # ?!
    #     'familyhistory_mother',  # ?!'
    #     'familyhistory_father',  # ?!
    #     'familyhistory_sibling1',  # ?!
    #     'familyhistory_sibling2',  # ?!
    #     'familyhistory_sibling3',  # ?!
    #     'familyhistory_sibling4',  # ?!
    #     'household_peoplelivingtogether',  # -> nicht numerisch!
    #     'household_otherolderchildren'  # nicht numerisch !
    # ]

    include = False
    include_exclude_columns = [
        'PID',
        'scorad',
        'stool_katokatz',
        'diagnosis_location',
        'RNASeq',
        'location',
        'eczema_age',
        'eczema_ever',
        'eczema_whodiagnosed',
        'medication_steroidcreams',
        'BSA'
    ]

elif subset == "Switzerland":
    include = True
    include_exclude_columns = [
        'paracetamol_exposure',
        'paracetamol_exposure_age',
        'paracetamol_exposure_time',
        'childhood_inf_other',
        'antibiotic_exposure',
        'antibiotic_exposure_age',
        'antibiotic_exposure_last2months',
        'sunlight_exp_winter',
        'sunlight_exp_summer',
        'cowmilkformula_exposure',
        'cowmilkformula_exposure_age_first',
        'cowmilkformula_exposure_regularity'
    ]
else:  # 'both'
    include = True
    include_exclude_columns = [
        'country',
        'paracetamol_exposure',
        'paracetamol_exposure_age',
        'paracetamol_exposure_time',
        'childhood_inf_mumps',
        'childhood_inf_chickenpox',
        'hildhood_inf_giandular_fever',
        'childhood_inf_tuberculosis',
        'childhood_inf_other',
        'antibiotic_exposure',
        'antibiotic_exposure_age',
        # 'antibiotic_exposure_courses',  # remove 0 from swiss children in raw data (categorical) -> does not work in explainerdashboard (reason: unknown...)
        'antibiotic_exposure_last2months',
        'anti_helminthic_exposure',
        'anti_helminthic_exposure_age',
        'anti_helminthic_exposure_last',
        'anti_helminthic_exposure_yearly',
        'sunlight_exp_winter',
        'sunlight_exp_summer',
        'cowmilkformula_exposure',
        'cowmilkformula_exposure_age_first',
        'cowmilkformula_exposure_regularity',
        'medicalhistory_comorbidities',
        'foodreaction_any',
        'familyhistory_mother_eczema',
        'familyhistory_mother_other_allergic_disease',
        'familyhistory_sibling_eczema',
        'familyhistory_sibling_other_allergic_disease',
        'fuel_cooking_Open fires in the house',
        'fuel_cooking_Paraffin Stove',
        'fuel_cooking_Electricity_Gas',
        'fuel_cooking_Open fires outside the house',
        'fuel_heating_Gas',
        'fuel_heating_Electricity',
        'fuel_heating_Kerosene_Paraffin',
        'fuel_heating_Wood_coal']

cv = [
    # ShuffleSplit(n_splits=5, test_size=0.2, random_state=2),
    ShuffleSplit(n_splits=5, test_size=0.3, random_state=8)
    # ShuffleSplit(n_splits=3, test_size=0.2, random_state=2),
    # ShuffleSplit(n_splits=3, test_size=0.3, random_state=8)
]

# PIPELINE FOR FITTING
pipe = Pipeline(steps=[('prep',
                        ColumnTransformer(transformers=[('num_pipeline',
                                                         Pipeline([
                                                             ('si', SimpleImputer()),
                                                             ('min_max', MinMaxScaler())
                                                         ]),
                                                         make_column_selector(dtype_include=np.number)
                                                         ),

                                                        ('cat_pipeline',
                                                         Pipeline([
                                                             ('imputer',
                                                              SimpleImputer(strategy="most_frequent")),
                                                             ('cfe',
                                                              CountFrequencyEncoder(encoding_method="frequency")),
                                                         ]),
                                                         make_column_selector(dtype_exclude=np.number)
                                                         )
                                                        ],
                                          remainder='drop',
                                          n_jobs=-1)),
                       ('corr', SmartCorrelatedSelection()),
                       ('constant', DropConstantFeatures()),
                       ])

# PARAMETERS FOR PIPELINE
parameters = (
    [
        {
            'prep__num_pipeline__si__strategy': ['mean', 'median'],
            'corr__threshold': [0.8, 0.9],  # drop features with values above threshold
            'constant__tol': [0.8, 0.9],  # threshold to drop features due low variance (x% of values are same)
            # 'corr__threshold': [0.7, 0.8, 0.9, 0.99],  # drop features with values above threshold
            # 'constant__tol': [0.8, 0.9, 0.95, 0.99],  # threshold to drop features due low variance (x% of values are same)
        }
    ],
    [
        {
            'estimator': [KNeighborsClassifier()],
            'estimator__n_neighbors': [5, 7, 9]
        },

        # {
        #     'estimator': [LogisticRegression()],
        #     'estimator__C': [0.1, 1, 10],
        #     'estimator__penalty': ['l2']
        # },
        {
            'estimator': [RandomForestClassifier()],
            # 'estimator__n_estimators': [50, 100],
            'estimator__n_estimators': [100, 200],
            'estimator__max_depth': [3, 4],
            # 'estimator__min_samples_split': [4, 6, 10],
            'estimator__min_samples_split': [4, 10],
            'estimator__random_state': [24]
            # 'estimator__max_depth': [1, 3, 5],
            # 'estimator__min_samples_split': [2, 4, 6, 10]
        },

        {
            'estimator': [AdaBoostClassifier()],
            'estimator__learning_rate': [0.1, 0.5, 1]
        },

        # {'estimator': [VotingClassifier(
        #     estimators=[('lr', LogisticRegression(random_state=1)),
        #                 ('rf', RandomForestClassifier(random_state=1)),
        #                 ('gnb', GaussianNB())]
        # )],
        #     'estimator__lr__C': [2.0, 100.0],
        #     # 'estimator__rf__n_estimators': [50, 100],
        #     'estimator__rf__n_estimators': [100],
        #     'estimator__rf__max_depth': [3, 4],
        #     # 'estimator__rf__min_samples_split': [4, 6, 10],
        #     'estimator__rf__min_samples_split': [4],
        #     # 'estimator__rf__max_depth': [1, 3, 5],
        #     # 'estimator__rf__min_samples_split': [2, 4, 10],
        #     'estimator__voting': ['soft'],  # TODO: hard -> no predict_proba()
        #     'estimator__weights': [[1, 1, 1], [3, 2, 1], [1, 2, 3]]
        # },
    ])

# *********************************************************************************************************************
# ************************************ EVALUATE_BUILD_ML_MODELS *******************************************************
# *********************************************************************************************************************

evaluate_ml_models_path = {
    "output_folder": f"{prefix}data/output/ml_models_{unique_key}/"  # store_path
}

# *********************************************************************************************************************
# *****************************************  EXPLAINER DASHBOARD ******************************************************
# *********************************************************************************************************************

explainer_dashboard_path = {
    "input_folder": f"{prefix}data/output/ml_models_{unique_key}",
    "output_folder": f"{prefix}data/output/ml_models_{unique_key}/dashboards"
}

recalculate = False  # dashboard will be recalculated even if it already exists in that folder if True
calculate_model_ids = ['0-0-7', '0-2-7']  # list of ids or a single id as string to calculate ExplainerDashboard

# run_dashboard = ['0-3-16', '1-0-2']  # id of desired model or False
run_dashboard = "all"

# these arguments are optional, of one, multiple or even None arguments are set, default values will be used
# automatically
explainer_args = {
    'cv': 5,
    'n_jobs': -1,
    'port': 8068
}
