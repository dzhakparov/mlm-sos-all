# Questions

## Preprocessing (file_transformation.ipynb)

* food_reaction_any: OKAY
* immun_number: OKAY
* family_history: OKAY
* fuel_cooking/heating: OKAY
* **sp_any** is missing -> sp_egg_white, sp_cowmilk, sp_soy, sp_wheet, sp_fish, sp_peanut 
  sp_hazelnut, sp_fresh_peanut, sp_fresh_egg, sp_fresh_cowmilk
  * can't build sp_any because but sp_neg and sp_pos are also missing! 
  * how to handle in ml-process? (True -> one feature (sp_any), False -> all sp_ features separate)

* **antibiotic_exposure_courses**: min: 1, max: 4 -> is feature categorical or numerical?
* **eczema_whodiagnosed**: min: 1, max:3 -> categorical?
* **parental_education_level**: min: 1, max:18 -> categorical?
* **solidfood_which_first, medication_oral_which**: only nan


* **Swiss children** are not in ```DataTransformed.txt``` -> they should be added in a second file
  with the same methods as the South African children (or in a second file combined with the South African children)
  * ATTENTION: **40010021** and  **40010031** appearing twice in raw data (SOSALL Master clinical data_NL_ms1_kb0.csv)!