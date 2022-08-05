# Questions / remarks

* From technically point of view the process is running... But there are plenty of points to decide to reach a 
reproducible und valide score at the end :-(
* build a container (apptainer, see Docker) to run all code and notebooks 
* all code shared on [Github](https://github.com/dzhakparov/mlm-sos-all)

## 1. file_transformation.ipynb (Preprocessing)

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

* CA-028-PN, CA-161-OM, CA-156-MV, CA-135-NS, CA-136_OM, CA-126-KB: why are there '.' as values for 'sp-...'-features?
* CO-010-OB: peanuts_mother_exposure_regularity and peanuts_mother_exposure_avoidance -> 'no' instead of 0
* why is 'stool-katokatz' not converted to numeric type?

* **Swiss children** are not in ```DataTransformed.txt``` -> they should be added in a second file
  with the same methods as the South African children (or in a second file combined with the South African children)
  * ATTENTION: **40010021** and  **40010031** appearing twice in raw data (SOSALL Master clinical data_NL_ms1_kb0.csv)!

## 2. statistical_tests.ipynb

* everything seems okay (results exists), no questions

## 3. build_ml_models.py

process is running technically, but
* there are +500 models at the end -> **which is the best?** -> highest cv-score? smallest gap between train and validation score? simplest model? ...
  * cross-validated (no test-data because there are too few data in general)
  * a lot of the models (AdaBoost) have (same) scores close to 1 -> remove features (like medication_steroidcreams, eczema_whodiagnosed)!
  * general: **108 features** against **217 observations...** ([Curse_of_dimensionality](https://en.m.wikipedia.org/wiki/Curse_of_dimensionality)) 
    * rule of thumbs: *"A typical rule of thumb is that there should be at least 5 training examples for each dimension in the representation"*
    * include 'SmartCorrelatedFeatures' (0.7, 0.8, 0.9, 0.99) and 'DropConstantFeatures' (0.8, 0.9, 0.95, 0.99) -> how many features dropped?
    * other **'Feature-Selection-Methods'** or **feature-engineering manually**?
* **general methodology:** train-, test-, validation-data -> chose 'best' model -> feature importance

## 4. evaluate_ml_models.ipynb

* works, but is not so feasible to find best model... Idea is to create an interactive solution with Dash 
  (with tables and plots) to discover models interactively 

## 5. explainer_dashboard.py

* works with ExplainerHub to 'compare' (or watch) several models