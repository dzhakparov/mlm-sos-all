# Protective and Susceptibility Clusters of Environmental Factors, Gene Expression, Antibody Responses, and Cytokines in Pediatric Atopic Dermatitis: Insights from Multi-Modal Data Integration

## Introduction 

This repository contains the analysis script for the publication {"Protective and Susceptibility Clusters of Environmental Factors, Gene Expression, Antibody Responses, and Cytokines in Pediatric Atopic Dermatitis: Insights from Multi-Modal Data Integration"}[https://www.medrxiv.org/content/10.64898/2026.01.10.26343854v1]. 

## Repository Structure 
 
`01_clinical_data/` - Contains the analysis file for the clinical data analysis
    * Code is intended to be run from an `Apptainer` container to use all resources from the underlying server or system and to be shareable and executable on all systems. See details in the Apptainer-part of the help-section (Sphinx)
    * **Information for PyCharm Users**: If the paths to the files in folder 'src' and to config.py are not shown properly (red -> can't reference to them)
        * **delete content root**: File -> settings -> Project: sosall -> Project structure
        * **add content root**: File -> settings -> Project: sosall -> Project structure -> **+** '/home/schmidmarco/Documents/CODE/PROJECTS/sosall/data_preperation'
    * **More Information**: More information is available under the Sphinx documentation
* **'sosall/docs/build/html/index.html'**

`02_data_integration/` - Data Integration scripts with DIABLO with additional files for the data preparation. 

`03_supplementary/` - Contains additional evaluation scripts for the working model (M52) and other supplementary material analyses listed in the paper. 

## Cloning the repository 
The project can be cloned from github.com with:
```
git clone https://github.com/dzhakparov/mlm-sos-all.git
```