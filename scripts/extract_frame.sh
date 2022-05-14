# Small research on this
# See here
# https://groups.inf.ed.ac.uk/vision/DATASETS/FISH4KNOWLEDGE/WEBSITE/F4KDATASAMPLES/CLIPS/
# Text explains to use the matlab function to extract frames.
# Text says that they use ffmpeg for fast extraction,
# but use Matlab VideoReeader, because ffmpeg is inexact.
# This is TERRIBLY SLOW, (full vid read into a Matlab memory object)
# text speaks about an hour per vid, (and I don't want to install matlab)
# Look into extractFrames.m
# They use -ss timestamp to offset ffmpeg
# Manpage tells -ss is inexact
# Online info tells the exact way to extract frames.
# https://stackoverflow.com/questions/38253406/extract-list-of-specific-frames-using-ffmpeg
# VideoReader reads into a matlab dataframe, this is 1-indexed.
# Thus frame_id in data must also be 1-index
# ffmpeg uses 0-indexing.
# must extract 1 from frame_id before feeding to ffmpeg
VID='./vids/'
DAT='./dat/'
FRAMES='./frames/'
LOG='./log'

VID_PATH=$1

VID_F=${VID_PATH##*/video_}
ID=${VID_F%%.flv}
DAT_PATH=${DAT}data_${ID}.csv

if [ ! -f ${DAT_PATH} ]; then
    echo ${ID} does not exists >> ${LOG}
    exit 1
fi

OUT="${FRAMES}${ID}"
FISH=${OUT}/fish/frame
NOFISH=${OUT}/nofish/frame
mkdir -p ${OUT}/{fish,nofish}

FRAME_IDS=${OUT}/frame_ids
NOFISH_FRAME_IDS=${OUT}/nofish_frame_ids
ALL_FRAME_IDS=${OUT}/all_frame_ids

tail +2 ${DAT_PATH} | awk '{print $6}' | tr -d , | sort -n -u > ${FRAME_IDS}
seq $(tail -1 "${FRAME_IDS}") | sort | comm -23 - <(sort "${FRAME_IDS}") |
    shuf | head -n $(wc -l < ${FRAME_IDS}) > ${NOFISH_FRAME_IDS}
sort -n ${FRAME_IDS} ${NOFISH_FRAME_IDS} > ${ALL_FRAME_IDS}
FFMPEG_EXTRACTION=$(
awk '{print "eq(n\\,"$1-1")"}' ${ALL_FRAME_IDS} |  # convert from matlab 1-indexed to ffmpeg 0-indexed
    paste -sd+  # concatenate into one string to interpret for ffmpeg
)
echo $FFMPEG_EXTRACTION

ffmpeg -i ${VID_PATH} -vf select="${FFMPEG_EXTRACTION}" -vsync -0 ${FISH}tmp%d.jpg
echo done ffmpg
nl ${ALL_FRAME_IDS} | # ffmpeg just outputs the frames numbered 1, this connect ffmpeg name with frame_id
    awk -v fish=${FISH} '{print fish"tmp"$1".jpg "fish$2".jpg"}' | # let's rename that
    xargs -n2 mv
echo done renaming
# now rename the nofish frame ids
awk -v nofish=${NOFISH} -v fish=${FISH} '{print fish$1".jpg "nofish$1".jpg"}' ${NOFISH_FRAME_IDS} |
    xargs -n2 mv
echo done moving

