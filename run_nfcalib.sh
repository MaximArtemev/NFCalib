#!/bin/bash
set -x
cd src && python nfcalib.py "$@"
