#!/bin/bash
# quickly and conveniently share changes to environments
# to enable working from different machines
# this allows you to do this running just one command instead of multiple ones

# export environment to a file
# include both conda and pip packages for complete environment reproduction
conda env export -n mlops-zoomcamp --no-builds > environment.yaml

# remove the prefix section
# it just shows the current local path of installation
# it won't work on other machines and is not needed to reproduce the environment
sed -i.bak '/^prefix:/d' environment.yaml && rm environment.yaml.bak

# track the updated environment file and push it to github
git add environment.yaml
git commit -m "Updated environment file with newly added dependencies."
git push
