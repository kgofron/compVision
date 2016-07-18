#!/usr/bin/env bash

printf "Retriving Tree..."
git diff --stat `git hash-object -t tree /dev/null` > tree.txt
