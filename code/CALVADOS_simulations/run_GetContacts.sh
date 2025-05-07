#!/bin/bash
source /path/to/miniconda/bin/activate getcontacts

#
topology="$1"
trajectory="$2"
output="$3"
python /path/to/getcontacts/get_dynamic_contacts.py --topology "$topology" \
 --trajectory "$trajectory" --itypes all --output "$output"

