URL_ROOT=https://groups.inf.ed.ac.uk/vision/DATASETS/FISH4KNOWLEDGE/WEBSITE/F4KDATASAMPLES/VIDEOS/FULLDAY/
HTML_FILE=vid.html
OUTPUT_FOLDER=./vids/

grep -o 'video.*flv' ${HTML_FILE} | xargs -t -P 5 -I{} wget -O ${OUTPUT_FOLDER}{} ${URL_ROOT}{}
