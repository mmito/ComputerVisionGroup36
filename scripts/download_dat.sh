URL_ROOT=https://groups.inf.ed.ac.uk/vision/DATASETS/FISH4KNOWLEDGE/WEBSITE/F4KDATASAMPLES/SQL/FULLDAY/
HTML_FILE=dat.html
OUTPUT_FOLDER=./dat/

grep -o 'data.*csv' ${HTML_FILE} | xargs -t -P 5 -I{} wget -O ${OUTPUT_FOLDER}{} ${URL_ROOT}{}
