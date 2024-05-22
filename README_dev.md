# AMPL API Documentation generation

## UML Class Diagram - PyReverse and Graphviz

Install PyReverse (part of pylint), and Graphviz for automatic class diagram creation

```shell
conda activate pipeline
conda install -c conda-forge pylint python-graphviz
# change directory into the pipeline/pacakge directory then run the following commands
mkdir docs/images
pyreverse src/ampl -p AMPL -d docs/images -o png
```

## Documentation creation - Sphinx

To run Sphinx on the project to create the documentation

Quickstart creates Sphinx folder structure and required files 

```shell
sphinx-apidoc -o docs src/ampl 
cd docs
make html
``` 

### Sphinx Themes

Example installing a theme
```pip install cloud_sptheme```

To use the them modify `conf.py` to use the cloud theme
`html_theme = 'cloud'`

### Additional Sphinx documentation (For reference only)

If you need to create a sphinx document folder then run the following command 
```commandline
cd ampl
sphinx-quickstart docs
```

For viewing the sphinx documentation, navigate to within the docs folder of the project. 

On windows run the following command
    `start _build/html/index.html`
On mac/Linux run the following command
    `open _build/html/index.html`

A quickstart guide on how to set and and run sphinx from scratch can be found here
    https://eikonomega.medium.com/getting-started-with-sphinx-autodoc-part-1-2cebbbca5365

Sphinx themes - a gallery of available themes
https://sphinx-themes.org/


# Additional Resources

**Git repo references** <br>
https://github.com/matiassingers/awesome-readme <br>
https://github.com/amitmerchant1990/electron-markdownify#readme <br>
https://github.com/anfederico/Clairvoyant#readme

**Quick references for markdown** <br>
https://www.markdownguide.org/basic-syntax/ <br>
https://markdown-it.github.io/ <br>
https://docs.gitlab.com/ee/user/markdown.html#table-of-contents <br>