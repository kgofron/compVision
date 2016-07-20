#!/usr/bin/env bash

# Writes tree to tree.txt and displays it on console
printf "Retriving Tree...\n"
git diff --stat `git hash-object -t tree /dev/null` > tree.txt
git diff --stat `git hash-object -t tree /dev/null`
printf "Finished writing to tree.txt\n"
