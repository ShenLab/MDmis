#!/bin/bash
source /path/to/miniconda/bin/activate cg2all

#
topology="$1"
trajectory="$2"
output_traj="$3"
output_pdb="$4"
convert_cg2all -p "$topology" \
 -d "$trajectory" -o "$output_traj" -opdb "$output_pdb" \
 --cg CalphaBasedModel

