#!/usr/bin/env bash

git pull
git log --graph --abbrev-commit --decorate --date=relative --all > gitlog.txt
git diff --stat `git hash-object -t tree /dev/null` > tree.txt
pydoc -w cvlib 
git add -A
git status
now=$(date +"%T - %D")
git commit -m "Update Files In Case Of Fire at $now"
git push
