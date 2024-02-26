rootpath=/vireo00/nikki/AVS_data
etime=1.0

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 testCollection score_file edition"
    exit
fi

test_collection=$1
score_file=$2
edition=$3
topk=$4
overwrite=$5

echo python txt2xml.py $test_collection $score_file --edition $edition --topk $topk --priority 1 --etime $etime --desc "This run uses the top secret x-component" --rootpath $rootpath --overwrite $overwrite

python txt2xml.py $test_collection $score_file --edition $edition --topk $topk --priority 1 --etime $etime --desc "This run uses the top secret x-component" --rootpath $rootpath --overwrite $overwrite


