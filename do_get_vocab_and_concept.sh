collection=$1
rootpath=/vireo00/nikki/AVS_data
threshold=5
overwrite=1

##step 1: obtain vocab for bow and RNN

for text_style in bow rnn
do
echo "python util/vocab.py $collection --rootpath $rootpath --threshold $threshold --text_style $text_style --overwrite $overwrite"
python util/vocab.py $collection --rootpath $rootpath --threshold $threshold --text_style $text_style --overwrite $overwrite
done

##step 2: obtain concept and contrary relations

echo "python build_concept.py $collection --rootpath $rootpath --threshold $threshold --overwrite $overwrite"
python build_concept.py $collection --rootpath $rootpath --threshold $threshold --overwrite $overwrite

echo "python detect_contrary_relation_wordnet.py $collection --rootpath $rootpath --threshold $threshold --overwrite $overwrite"
python detect_contrary_relation_wordnet.py $collection --rootpath $rootpath --threshold $threshold --overwrite $overwrite

echo "python readContractPairs.py $collection --rootpath $rootpath --threshold $threshold --overwrite $overwrite"
python readContractPairs.py $collection --rootpath $rootpath --threshold $threshold --overwrite $overwrite

