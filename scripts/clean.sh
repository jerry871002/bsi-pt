#!/bin/zsh

# remove logs
printf "Removing logs...\n"
ls **/*.log
rm **/*.log

# remove data
printf "\nRemoving data...\n"
ls **/*.pkl
rm **/*.pkl

# remove csv
printf "\nRemoving csv...\n"
ls **/*.csv
rm **/*.csv