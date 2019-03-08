#!/usr/bin/env bash
original_path=$PWD
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"
cd $SCRIPTPATH
git rev-parse --verify HEAD
cd $original_path