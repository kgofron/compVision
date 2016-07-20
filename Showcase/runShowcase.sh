#!/usr/bin/env bash

printf "Running Showcase...\n"

python BPM1Proc.py
printf "\n\n"
python merlin.py
printf "\n\n"
python matchTest.py
printf "\n\n"
python mountBeam.py
printf "\n\nEnding Showcase...\n"
