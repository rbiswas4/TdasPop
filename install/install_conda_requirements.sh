conda update
conda install -c anaconda --yes --file ./install/conda_requirements_anaconda.txt
conda install -c conda-forge --yes --file ./install/conda_requirements_conda_forge.txt
conda list --explicit > ./install/spec-file.txt;
