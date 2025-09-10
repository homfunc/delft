#!/usr/bin/env bash
# Usage: source scripts/enable_left_padding.sh
# Sets an environment variable that instructs DeLFT's CRF generator to use left padding masks (valid tokens right-aligned).
export DELFT_CRF_LEFT_PADDING=1
printf "DELFT_CRF_LEFT_PADDING=1 (left padding enabled)\n"

