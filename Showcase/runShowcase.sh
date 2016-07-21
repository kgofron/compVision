#!/usr/bin/env bash

printf "Running Showcase...\n"

time python BPM1Proc.py
printf "\n\n"
time python merlin.py
printf "\n\n"
time python matchTest.py
printf "\n\n"
time python mountBeam.py
printf "\n\nEnding Showcase...\n"
