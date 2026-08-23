#!/usr/bin/env bash

DEST="markdown"
rm -rf "$DEST"
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/your5e/5e-srd-markdown.git "$DEST"
git -C "$DEST" sparse-checkout set "dnd/521/markdown"
mv $DEST/dnd/521/markdown/* $DEST
rm -rf $DEST/dnd
rm -rf "$DEST/.git"
touch "$DEST/.gitkeep"
echo "OK: $(find "$DEST" -name '*.md' | wc -l) file"