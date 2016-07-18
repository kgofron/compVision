#!/usr/bin/env bash

printf "TOTAL WORD COUNT:\n"
git ls-files | xargs wc -l
printf "\nPython Files Word Count:\n"
git ls-files | grep '.py\|.sh\|.md\|.txt\|LICENSE' | grep -v '.pyc' | xargs wc -l

