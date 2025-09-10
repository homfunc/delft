#!/usr/bin/env bash
# Usage: source scripts/enable_right_padding.sh
# Unsets the env var to use right padding masks (valid tokens left-aligned), which is the default.
unset DELFT_CRF_LEFT_PADDING
printf "DELFT_CRF_LEFT_PADDING unset (right padding enabled)\n"

