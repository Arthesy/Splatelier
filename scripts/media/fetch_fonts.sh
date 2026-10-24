#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Downloads the fonts the animation scripts draw with into scripts/media/fonts/ (not committed).
# All three are licensed under the SIL Open Font License 1.1; the license texts are saved next to
# the font files. Idempotent: files that are already present are kept.
#
#   Instrument Serif  the "Splatelier" wordmark of the launch animation (same as the logo)
#   Lora              titles of the capture guide clips
#   Noto Sans JP      captions and labels of the capture guide clips

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FONT_DIR="$SCRIPT_DIR/fonts"
BASE="https://raw.githubusercontent.com/google/fonts/main/ofl"
mkdir -p "$FONT_DIR"

fetch() { # <url path under ofl/> <output name>
    if [ -s "$FONT_DIR/$2" ]; then
        echo "have $2"
    else
        echo "fetching $2"
        curl -fsSL --retry 3 --max-time 300 -o "$FONT_DIR/$2" "$BASE/$1"
    fi
}

fetch "instrumentserif/InstrumentSerif-Regular.ttf" "InstrumentSerif-Regular.ttf"
fetch "instrumentserif/OFL.txt" "OFL-InstrumentSerif.txt"
fetch "lora/Lora%5Bwght%5D.ttf" "Lora-VF.ttf"
fetch "lora/OFL.txt" "OFL-Lora.txt"
fetch "notosansjp/NotoSansJP%5Bwght%5D.ttf" "NotoSansJP-VF.ttf"
fetch "notosansjp/OFL.txt" "OFL-NotoSansJP.txt"
echo "fonts in $FONT_DIR"
