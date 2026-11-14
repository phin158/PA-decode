#!/bin/bash
# Clone the authors' ASOT code at the commit used in the paper and install the Python dependencies.
set -e
ASOT_DIR=third_party/action_seg_ot
COMMIT=0c4c86b1
if [ ! -d "$ASOT_DIR" ]; then
    git clone -q https://github.com/mingu6/action_seg_ot.git "$ASOT_DIR"
fi
git -C "$ASOT_DIR" checkout -q "$COMMIT"
echo "ASOT at $(git -C "$ASOT_DIR" rev-parse --short HEAD)"
if [ "$1" != "--no-pip" ]; then
    python -m pip install -r requirements.txt
fi
