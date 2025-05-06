#!/bin/bash
source /home/az2798/miniconda3/bin/activate getcontacts

#
topology="$1"
trajectory="$2"
output="$3"
python /home/az2798/getcontacts/get_dynamic_contacts.py --topology "$topology" \
 --trajectory "$trajectory" --itypes all --output "$output"

