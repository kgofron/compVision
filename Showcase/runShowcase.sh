#!/usr/bin/env bash

printf "Running Showcase...\n"

time python BPM1ProcNoDis.py
printf "\n\n"
time python merlinNoDis.py
printf "\n\n"
time python matchTestNoDis.py
printf "\n\n"
time python mountBeamNoDis.py
printf "\n\nEnding Showcase...\n"
