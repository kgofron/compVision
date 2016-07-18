#!/usr/bin/env bash

printf "Retriving Tree...\n"
git diff --stat `git hash-object -t tree /dev/null` > tree.txt
