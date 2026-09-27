#!/bin/bash

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -f|--file) file="$2"; shift ;;
        -n|--name) name="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Run script
tmux new-session -d -s "$name" \
    bash -lc "conda activate pofft && python \"$file\""