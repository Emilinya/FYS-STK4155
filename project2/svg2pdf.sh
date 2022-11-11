#!/bin/sh

# simple script to convert svgs to pdfs that can be used with latex
FILES="./imgs/misc/*.svg ./imgs/linfit/*.svg ./imgs/ffnn_regression/*.svg ./imgs/ffnn_logreg/*.svg ./imgs/ffnn_classification/*.svg"
for f in $FILES
do
  echo "Processing $f..."
  NEW_NAME=$(echo $(echo "$f" | sed -e 's/svg/pdf/g') | sed -e 's/imgs/imgs\/pdfimgs/g')
  NEW_FOLDER=$(echo "$NEW_NAME" | sed -e 's/\/\w*\.pdf//g')

  mkdir -p $NEW_FOLDER
  rsvg-convert -f pdf -o $NEW_NAME $f
done
