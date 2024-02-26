##script to train the ITV model

trainCollection=tgif-msrvtt10k-VATEX
valCollection=tv2016train
testCollection=iacc.3
n_caption=2

rootpath=/vireo00/nikki/AVS_data
visual_feature=pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os
motion_feature=mean_slowfast+mean_swintrans

lr=0.0002
overwrite=1
epoch=100
direction=all
cost_style=sum
lambda=0.2
ul_alpha=0.01
decoder_layers=0-2048
classification_loss_type=favorBCEloss  ##favorBCEloss|normalBCEloss
concept_fre_threshold=5
concept_bank=concept_word

postfix=run_ITV_on_${trainCollection}
echo "CUDA_VISIBLE_DEVICES=$gpu python train_ITV.py $trainCollection $valCollection $testCollection --rootpath $rootpath --overwrite $overwrite \
	 --concept_bank ${concept_bank}  --concept_fre_threshold $concept_fre_threshold --unlikelihood  --decoder_layers $decoder_layers --motion_feature $motion_feature --multiclass_loss_lamda ${lambda}  --ul_alpha ${ul_alpha} --max_violation --learning_rate $lr --num_epochs $epoch --text_norm --visual_norm --visual_feature $visual_feature --n_caption $n_caption --direction $direction --postfix $postfix --cost_style $cost_style > output/$postfix.out"

CUDA_VISIBLE_DEVICES=$gpu python train_ITV.py $trainCollection $valCollection $testCollection --rootpath $rootpath --overwrite $overwrite \
	 --concept_bank ${concept_bank}  --concept_fre_threshold $concept_fre_threshold --unlikelihood  --decoder_layers $decoder_layers --motion_feature $motion_feature --multiclass_loss_lamda ${lambda}  --ul_alpha ${ul_alpha} --max_violation --learning_rate $lr --num_epochs $epoch --text_norm --visual_norm --visual_feature $visual_feature --n_caption $n_caption --direction $direction --postfix $postfix --cost_style $cost_style > output/$postfix.out



