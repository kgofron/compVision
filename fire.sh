#!/usr/bin/env bash

git add -A
now=$(date +"%T - %D")
git commit -m "Update Files In Case Of Fire at $now"
git push
