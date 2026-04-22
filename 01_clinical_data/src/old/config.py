from datetime import datetime

dt_now = datetime.now().strftime('%Y-%m-%d_%H:%M')

#  ****************** GLOBAL_PREPROCESSING ******************

target = 'diagnosis'  # diagnosis_location, diagnosis
subset = 'South Africa'  # ('South Africa' , 'Switzerland', 'both') # both: african kids as train and swiss kids as testset
# -> does only concern statistical_tests and eda but not the production of final_data.pkl and final_data.csv
sp_any = True  # True -> one feature (sp_any), False -> all sp_ features separate

input_folder = "data/input"
output_folder = f"data/output"

raw_data_file = f"{input_folder}/SOSALL Master clinical data_NL_ms1_kb0.csv"
specification_file = f"{input_folder}/description_data_ms8_kb2.csv"  # Änderung: Version!
mapping = f'{input_folder}/mapping_metadatanames_pid_def.csv'
metadata = f'{input_folder}/metadata.txt'

mv = [["n/a", "na", "???", "?", ".", "NR", ""]]  # missing values in file
print_reports = True
