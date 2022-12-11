#!/bin/sh

# simple script to convert svgs to pdfs that can be used with latex
FILES="./imgs/ffnn_solver/*.svg ./imgs/num_solver/*.svg ./imgs/pytorch_solver/*.svg  ./imgs/poly_solver/*.svg ./imgs/multi/*.svg "
for f in $FILES
do
    echo "Processing $f..."
    NEW_NAME=$(echo $(echo "$f" | sed -e 's/svg/pdf/g') | sed -e 's/imgs/imgs\/pdfimgs/g')
    NEW_FOLDER=$(echo "$NEW_NAME" | sed -e 's/\/[^\/]*\.pdf//g')

    mkdir -p $NEW_FOLDER
    rsvg-convert -f pdf -o $NEW_NAME $f
done
