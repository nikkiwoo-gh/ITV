rootpath=/vireo00/nikki/AVS_data
testCollection=v3c2
overwrite=1
topk=1000

for topic_set in {tv22,tv23}; do

    for theta in {0_0,0_3,0_5,1_0}
        do
            score_file=/vireo00/nikki/AVS_data/v3c2/results/${topic_set}.avs.txt/tgif-msrvtt10k-VATEX/tv2016train/ITV_word_only_dp_0.2_measure_cosine_lambda_0.2/vocab_word_vocab_5_word_dim_500_text_rnn_size_512_text_norm_True_kernel_sizes_2-3-4_num_512/visual_feature_pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os_visual_rnn_size_1024_visual_norm_True_kernel_sizes_2-3-4-5_num_512/mapping_text_0-2048_img_0-2048_decoder_0-2048/loss_func_mrl_margin_0.2_direction_all_max_violation_True_cost_style_sum/optimizer_adam_lr_0.0002_decay_0.99_grad_clip_2.0_val_metric_recall/run_ITV_on_v3c2/model_best.pth.match.tar/id.sent.sim.0.99.combinedDecodedConcept_theta${theta}_score

            bash do_txt2xml.sh $testCollection $score_file $topic_set $topk $overwrite
            echo python trec_eval.py ${score_file}.xml --topk $topk --rootpath $rootpath --collection $testCollection --edition $topic_set --overwrite $overwrite
            python trec_eval.py ${score_file}.xml --topk $topk --rootpath $rootpath --collection $testCollection --edition $topic_set --overwrite $overwrite

    done
done
