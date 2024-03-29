from __future__ import print_function

import pickle
import os
import sys
sys.path.append('./util')
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import json
import numpy as np
import torch
import evaluation
import util.data_provider as data
import logging
import time
import argparse
import wandb


from util.util import Progbar
from util.vocab import Vocabulary,Concept
from util.text2vec import get_text_encoder
from model import ITV, get_we_parameter
from util.bigfile import BigFile
from util.util import read_dict, AverageMeter, LogCollector,makedirsforfile,checkToSkip
from util.constant import ROOT_PATH


INFO = __file__

def parse_args():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--rootpath', type=str, default=ROOT_PATH,help='path to datasets')
    parser.add_argument('--savepath', type=str, default=ROOT_PATH, help='path to save.')
    parser.add_argument('trainCollection', type=str, help='train collection')
    parser.add_argument('valCollection', type=str,  help='validation collection')
    parser.add_argument('testCollection', type=str,  help='test collection')
    parser.add_argument('--n_caption', type=int, default=20, help='number of captions of each image/video (default: 1)')
    parser.add_argument('--overwrite', type=int, default=0, choices=[0,1], help='overwrite existed file. (default: 0)')
    parser.add_argument('--concept_bank', type=str, default='concept_word', help='concept_bank filename')
    parser.add_argument('--concept_fre_threshold', type=int, default=5, help='concept frequence threshold')
    # model
    parser.add_argument('--model', type=str, default='ITV', help='model name. (default: dual_task)')
    parser.add_argument('--vconcate', type=str, default='full',
                        help='visual feature concatenation style. (full|reduced) full=level 1+2+3; reduced=level 2+3')
    parser.add_argument('--tconcate', type=str, default='full',
                        help='textual feature concatenation style. (full|reduced) full=level 1+2+3; reduced=level 2+3')
    parser.add_argument('--measure', type=str, default='cosine', help='measure method. (default: cosine)')
    parser.add_argument('--dropout', default=0.2, type=float, help='dropout rate (default: 0.2)')
    # text-side multi-level encoding
    parser.add_argument('--vocab', type=str, default='word_vocab_5', help='word vocabulary. (default: word_vocab_5)')
    parser.add_argument('--word_dim', type=int, default=500, help='word embedding dimension')
    parser.add_argument('--text_rnn_size', type=int, default=512, help='text rnn encoder size. (default: 1024)')
    parser.add_argument('--text_kernel_num', default=512, type=int, help='number of each kind of text kernel')
    parser.add_argument('--text_kernel_sizes', default='2-3-4', type=str, help='dash-separated kernel size to use for text convolution')
    parser.add_argument('--text_norm', action='store_true', help='normalize the text embeddings at last layer')
    # video-side multi-level encoding
    parser.add_argument('--visual_feature', type=str, default='pyresnext-101_rbps13k,flatten0_output,os+pyresnet-152_imagenet11k,flatten0_output,os', help='visual feature.')
    parser.add_argument('--motion_feature', type=str, default='mean_slowfast+mean_swintrans', help='motion feature.')
    parser.add_argument('--visual_rnn_size', type=int, default=1024, help='visual rnn encoder size')
    parser.add_argument('--visual_kernel_num', default=512, type=int, help='number of each kind of visual kernel')
    parser.add_argument('--visual_kernel_sizes', default='2-3-4-5', type=str, help='dash-separated kernel size to use for visual convolution')
    parser.add_argument('--visual_norm', action='store_true', help='normalize the visual embeddings at last layer')
    ##unify_decoder
    parser.add_argument('--decoder_layers', type=str, default='0-2048', help='decoder FC layers.')
    # common space learning
    parser.add_argument('--text_mapping_layers', type=str, default='0-2048', help='text fully connected layers for common space learning. (default: 0-2048)')
    parser.add_argument('--visual_mapping_layers', type=str, default='0-2048', help='visual fully connected layers  for common space learning. (default: 0-2048)')
    # loss
    parser.add_argument('--loss_fun', type=str, default='mrl', help='loss function')
    parser.add_argument('--loss_type', type=str, default='favorBCEloss', help='loss function for the classification loss')
    parser.add_argument('--margin', type=float, default=0.2, help='rank loss margin')
    parser.add_argument('--direction', type=str, default='all', help='retrieval direction (all|t2i|i2t)')
    parser.add_argument('--max_violation', action='store_true', help='use max instead of sum in the rank loss')
    parser.add_argument('--cost_style', type=str, default='sum', help='cost style (sum, mean). (default: sum)')

    parser.add_argument('--decoder_loss_fun', type=str, default='BCEloss', help='loss function')
    parser.add_argument('--multiclass_loss_lamda', type=float, default='0.1', help='how many favor positive loss function')
    parser.add_argument('--ul_alpha', type=float, default='0.01',help='hyperparmeter in unlikelyhood training')
    # optimizer
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer. (default: rmsprop)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--decoder_learning_rate', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--lr_decay_rate', default=0.99, type=float, help='learning rate decay rate. (default: 0.99)')
    parser.add_argument('--grad_clip', type=float, default=2, help='gradient clipping threshold')
    parser.add_argument('--unlikelihood', action='store_true', help='use unlikelihood in training')
    parser.add_argument('--resume', action='store_true', help='use it to resume the model parameters')
    parser.add_argument('--resume_path',default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--checkpoint_name', default='model_best.pth.match.tar', type=str,
                        help='name of checkpoint (default: model_best.pth.tar)')
    parser.add_argument('--val_metric', default='recall', type=str, help='performance metric for validation (mir|recall)')
    # misc
    parser.add_argument('--num_epochs', default=100, type=int, help='Number of training epochs.')
    parser.add_argument('--decoder_num_epochs', default=30, type=int, help='Number of training epochs.')
    parser.add_argument('--batch_size', default=128, type=int, help='Size of a training mini-batch.')
    parser.add_argument('--workers', default=10, type=int, help='Number of data loader workers.')
    parser.add_argument('--postfix', default='runs_ITV', help='Path to save the model and Tensorboard log.')
    parser.add_argument('--log_step', default=10, type=int, help='Number of steps to print and record the log.')
    parser.add_argument('--cv_name', default='ACMTOIS', type=str, help='')
    parser.add_argument('--project_name', default='ITV', type=str, help='')

    args = parser.parse_args()
    return args


def main():
    opt = parse_args()
    print(json.dumps(vars(opt), indent = 2))

    rootpath = opt.rootpath
    trainCollection = opt.trainCollection
    valCollection = opt.valCollection
    testCollection = opt.testCollection
    opt.project_name = 'ITV_'+trainCollection
    wandb.init(project=opt.project_name)

    if opt.loss_fun == "mrl" and opt.measure == "cosine":
        assert opt.text_norm is True
        assert opt.visual_norm is True

    # checkpoint path
    model_info = '%s_word_only_dp_%.1f_measure_%s_lambda_%.1f' %  (opt.model, opt.dropout, opt.measure,opt.multiclass_loss_lamda)
    # text encoder info
    text_encode_info = 'vocab_%s_word_dim_%s_text_rnn_size_%s_text_norm_%s' % \
            (opt.vocab, opt.word_dim, opt.text_rnn_size, opt.text_norm)
    text_encode_info += "_kernel_sizes_%s_num_%s" % (opt.text_kernel_sizes, opt.text_kernel_num)
    #video encoder encoding info
    visual_encode_info = 'visual_feature_%s_visual_rnn_size_%d_visual_norm_%s' % \
            (opt.visual_feature, opt.visual_rnn_size, opt.visual_norm)
    visual_encode_info += "_kernel_sizes_%s_num_%s" % (opt.visual_kernel_sizes, opt.visual_kernel_num)
    # joint space learning info
    mapping_info = "mapping_text_%s_img_%s_decoder_%s" % (opt.text_mapping_layers, opt.visual_mapping_layers,opt.decoder_layers)
    loss_info = 'loss_func_%s_margin_%s_direction_%s_max_violation_%s_cost_style_%s' % \
                    (opt.loss_fun, opt.margin, opt.direction, opt.max_violation, opt.cost_style)
    optimizer_info = 'optimizer_%s_lr_%s_decay_%.2f_grad_clip_%.1f_val_metric_%s' % \
                    (opt.optimizer, opt.learning_rate, opt.lr_decay_rate, opt.grad_clip, opt.val_metric)

    opt.logger_name = os.path.join(opt.savepath, trainCollection, opt.cv_name, valCollection, model_info, text_encode_info,
                            visual_encode_info, mapping_info, loss_info, optimizer_info, opt.postfix)
    print(opt.logger_name)

    if checkToSkip(os.path.join(opt.logger_name, 'model_best.pth.match.tar'), opt.overwrite):
        sys.exit(0)
    if checkToSkip(os.path.join(opt.logger_name, 'val_metric.txt'), opt.overwrite):
        sys.exit(0)
    makedirsforfile(os.path.join(opt.logger_name, 'val_metric.txt'))
    logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)


    opt.text_kernel_sizes = list(map(int, opt.text_kernel_sizes.split('-')))
    opt.visual_kernel_sizes = list(map(int, opt.visual_kernel_sizes.split('-')))

    # collections: trian, val
    collections = {'train': trainCollection, 'val': valCollection}
    cap_file = {'train': trainCollection + '.caption.txt',
                'val': valCollection + '.caption.txt'}

    # caption
    caption_files = { x: os.path.join(rootpath, collections[x], 'TextData', cap_file[x])
                        for x in collections }

    # Load visual features
    visual_feat_path = {x: os.path.join(rootpath, collections[x], 'FeatureData', opt.visual_feature)
                        for x in collections }
    visual_feats = {x: BigFile(visual_feat_path[x]) for x in visual_feat_path}
    motion_feat_path = {x: os.path.join(rootpath, collections[x], 'FeatureData', opt.motion_feature)
                        for x in collections}
    motion_feats = {x: BigFile(motion_feat_path[x]) for x in motion_feat_path}
    opt.visual_feat_dim = visual_feats['train'].ndims
    opt.motion_feat_dim = motion_feats['train'].ndims

    bow_vocab_file = os.path.join(rootpath, opt.trainCollection, 'TextData', 'vocabulary', 'bow', opt.vocab+'.pkl')
    bow_vocab = pickle.load(open(bow_vocab_file, 'rb'))
    bow2vec = get_text_encoder('bow')(bow_vocab)
    opt.bow_vocab_size = len(bow_vocab)

    # set rnn vocabulary
    rnn_vocab_file = os.path.join(rootpath, opt.trainCollection, 'TextData', 'vocabulary', 'rnn', opt.vocab + '.pkl')
    rnn_vocab = pickle.load(open(rnn_vocab_file, 'rb'))
    opt.vocab_size = len(rnn_vocab)

    # initialize word embedding
    opt.we_parameter = None
    if opt.word_dim == 500:
        w2v_data_path = os.path.join(rootpath, "word2vec", 'flickr', 'vec500flickr30m')
        opt.we_parameter = get_we_parameter(rnn_vocab, w2v_data_path)

    # mapping layer structure
    opt.text_mapping_layers = list(map(int, opt.text_mapping_layers.split('-')))
    opt.visual_mapping_layers = list(map(int, opt.visual_mapping_layers.split('-')))
    # visual concatenation
    if opt.vconcate == 'full':  # level 1+2+3+4
        opt.visual_mapping_layers[0] = opt.visual_feat_dim + opt.visual_rnn_size * 2 + opt.visual_kernel_num * len(
            opt.visual_kernel_sizes)+opt.motion_feat_dim
    else:
        raise NotImplementedError('Model %s not implemented' % opt.model)
    # texutal concatenation
    if opt.tconcate == 'full':  # level 1+2+3
        opt.text_mapping_layers[0] = opt.bow_vocab_size + opt.text_rnn_size * 2 + opt.text_kernel_num * len(
            opt.text_kernel_sizes)
    else:
        raise NotImplementedError('Model %s not implemented' % opt.model)


    cap_prefixs = {'train': '.caption', 'val': '.caption'}
    concept_videolevel_paths = {
        'train': os.path.join(rootpath, collections['train'], 'TextData', collections['train'] + cap_prefixs[
            'train'] + '.txt.concept_videolevel.'+opt.concept_bank),
        'val': os.path.join(rootpath, collections['val'], 'TextData', collections['val'] + cap_prefixs[
            'val'] + '.txt.concept_videolevel.'+opt.concept_bank)
    }

    concept_file = os.path.join(rootpath, opt.trainCollection, 'TextData', 'concept', 'concept_frequency_count_gt' + str(
                                          opt.concept_fre_threshold)+'.'+opt.concept_bank+'.txt')

    opt.concept_file = concept_file
    print("concept file: %s"%concept_file)
    with open(concept_file, 'r') as reader:
        concept_lines = reader.readlines()
    ##create the concept structure
    concept= Concept()
    for iconcept in concept_lines:
        iconcept = iconcept.strip().split()
        concept.add_concept(' '.join(iconcept[0:-1]))

    ##add contraction pairs
    contradiction_file = concept_file+'.contradict.contradict_pairs'
    with open(contradiction_file, 'r') as reader:
        lines = reader.readlines()
    for line in lines:
        if line.find('//') < 0:
            concept.add_contradict(line)

    concept2vec = get_text_encoder('bow')(concept, istimes=0)
    opt.concept_list_size = len(concept)
    opt.concept = concept
    #construct contradicted matrix for training. Matirx is [len(concept_list)*len(concept_list)]
    contradicted_matrix_np = np.zeros([opt.concept_list_size, opt.concept_list_size])
    for key in concept2vec.vocab.idx2contractIdx.keys():
        values = concept2vec.vocab.idx2contractIdx[key]
        contradicted_matrix_np[key, values] = 1
    print("@contray pairs:%d"%contradicted_matrix_np.sum())
    contradicted_matrix_sp = torch.from_numpy(contradicted_matrix_np).to_sparse()
    if torch.cuda.is_available():
        contradicted_matrix_sp = contradicted_matrix_sp.cuda()
    opt.contradicted_matrix_sp = contradicted_matrix_sp
    del contradicted_matrix_np

    # Construct the model
    decoder_layers=opt.decoder_layers.split('-')
    del decoder_layers[0]
    decoder_layers.append(opt.concept_list_size)
    opt.decoder_mapping_layers = [int(ilayer) for ilayer in decoder_layers]
    model = ITV(opt)
    opt.we_parameter = None

    # set data loader
    video2frames = {x: read_dict(os.path.join(rootpath, collections[x], 'FeatureData', opt.visual_feature,'video2frames.txt'))
                    for x in collections }

    # optionally resume from a checkpoint to the encoder
    if opt.resume:
        resume_path = os.path.join(opt.resume_path, opt.checkpoint_name)
        if os.path.isfile(resume_path):
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch']
            matching_best_rsum = checkpoint['matching_best_rsum']
            classification_best_rsum = checkpoint['classification_best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another
            # training
            model.Eiters = checkpoint['classification_Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, matching_best_rsum {},classification_best_rsum {})"
                  .format(resume_path, start_epoch, matching_best_rsum, classification_best_rsum))
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))

    data_loaders = data.get_vid_txt_data_loaders(
    caption_files, visual_feats, motion_feats, rnn_vocab, bow2vec, concept2vec, opt.batch_size, opt.workers,
    opt.n_caption, video2frames=video2frames,concept_video_level=concept_videolevel_paths)

    # Train the Model
    best_matching_currscore = 0
    best_classification_currscore= 0
    no_impr_counter = 0
    lr_counter = 0
    best_epoch = None
    best_recall = 0
    matching_best_epoch = None
    classification_best_epoch = None
    fout_val_metric_hist = open(os.path.join(opt.logger_name, 'val_metric_hist.txt'), 'w')
    for epoch in range(opt.num_epochs):
        print('Epoch[{0} / {1}] R: {2}'.format(epoch, opt.num_epochs, get_learning_rate(model.optimizer)[0]))
        print('-'*10)
        # train for one epoch
        model.vid_encoder.train()
        model.text_encoder.train()
        model.unify_decoder.train()

        train_dual_task(opt, data_loaders['train'], model,epoch)
        model.vid_encoder.eval()
        model.text_encoder.eval()
        model.unify_decoder.eval()
        # evaluate on validation set
        matching_currscore, classification_vid_cur_recall,classification_text_cur_recall   = evaluation.eval_ITV(opt, data_loaders['val'], model,concept2vec,opt.measure)

        classification_currscore = (classification_vid_cur_recall+classification_text_cur_recall)/2
        # remember best R@ sum and save checkpoint
        matching_is_best = matching_currscore > best_matching_currscore
        classification_is_best = classification_currscore > best_classification_currscore
        if matching_is_best:
            matching_best_epoch = epoch
        if classification_is_best:
            classification_best_epoch = epoch

        best_matching_currscore = max(matching_currscore, best_matching_currscore)
        best_classification_currscore = max(classification_currscore, best_classification_currscore)
        print(' * matching Current perf: {}'.format(matching_currscore))
        print(' * matching Best perf: {}'.format(best_matching_currscore))
        print(' * classification Current perf: {}'.format(classification_currscore))
        print(' * classification Best perf: {}'.format(best_classification_currscore))
        print('')
        fout_val_metric_hist.write(
            'epoch_%d(matching,classification): %f,%f\n' % (epoch, matching_currscore, classification_currscore))
        fout_val_metric_hist.flush()

        # is_best = (matching_is_best | classification_is_best)
        is_best = matching_is_best
        if is_best:
            if matching_is_best:
                best_epoch = None
                filename = 'model_best.pth.match.tar'
                epoch_best_classification_score = classification_currscore
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model': model.state_dict(),
                    'matching_best_rsum': matching_currscore,
                    'classification_best_rsum': classification_currscore,
                    'opt': opt,
                    'classification_Eiters': model.Eiters,
                }, matching_is_best, filename=filename, prefix=opt.logger_name + '/',
                    best_epoch=best_epoch)
                best_epoch = epoch

        lr_counter += 1
        decay_learning_rate(opt, model.optimizer, opt.lr_decay_rate)
        early_stop=2
        if not is_best:
            # Early stop occurs if the validation performance does not improve in consecutive epochs
            # and loss does not decrease
            no_impr_counter += 1
            if (no_impr_counter > early_stop):
                print('Early stopping happended.\n')
                break

            # When the validation performance decreased after an epoch,
            # we divide the learning rate by 2 and continue training;
            # but we use each learning rate for at least 3 epochs.
            if lr_counter > 2:
                decay_learning_rate(opt, model.optimizer, 0.5)
                lr_counter = 0
        else:
            no_impr_counter = 0

    print('best performance on validation: matching{}\n'.format(best_matching_currscore))
    print('classification{}\n'.format(best_classification_currscore))
    print('@matching_best epoch:{}\n'.format(matching_best_epoch))
    print('@classification best epoch:{}\n'.format(classification_best_epoch))
    print('best_recall:{}\n'.format(best_recall))
    with open(os.path.join(opt.logger_name, 'val_metric.txt'), 'w') as fout:
        fout.write('best performance on validation: epoch ' + str(best_epoch)+
                   'matching' + str(best_matching_currscore) +
                   ', classification' + str(epoch_best_classification_score))

    fout_val_metric_hist.close()


def train_dual_task(opt, train_loader, model, epoch):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()



    progbar = Progbar(len(train_loader.dataset))
    end = time.time()
    for i, train_data in enumerate(train_loader):
        # video_data, text_data,concept_bows, caption_ori,idxs, cap_ids, video_ids= train_data
        # measure data loading time
        wandb.log({"train/lr": get_learning_rate(model.optimizer)[0]}, step=model.Eiters)

        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        if opt.unlikelihood:
            b_size, loss,loss_matching, likelihoodLoss,likelihoodLoss_vid, likelihoodLoss_text,unlikelihoodLoss, unlikelihoodLoss_vid,unlikelihoodLoss_text = model.train_dualtask(*train_data)
            progbar.add(b_size, values=[('loss', loss),('matching_loss', loss_matching),
                                        ('likelihoodLoss',likelihoodLoss),('unlikelihoodLoss', unlikelihoodLoss),
                                        ('likelihoodLoss_vid', likelihoodLoss_vid), ('likelihoodLoss_text', likelihoodLoss_text),
                                        ('unlikelihoodLoss_vid', unlikelihoodLoss_vid),('unlikelihoodLoss_text', unlikelihoodLoss_text)])
        else:
            b_size, loss,loss_matching, likelihoodLoss,likelihoodLoss_vid, likelihoodLoss_text = model.train_dualtask(*train_data)
            progbar.add(b_size, values=[('loss', loss),('matching_loss', loss_matching),
                                        ('likelihoodLoss',likelihoodLoss),
                                        ('likelihoodLoss_vid', likelihoodLoss_vid), ('likelihoodLoss_text', likelihoodLoss_text)])
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Record logs in wandb
        wandb.log({"train/eiters": epoch}, step=model.Eiters)
        wandb.log({"train/step": i}, step=model.Eiters)
        wandb.log({"train/batch_time": batch_time.val}, step=model.Eiters)
        wandb.log({"train/data_time": data_time.val}, step=model.Eiters)

        wandb.log({"train/loss": loss}, step=model.Eiters)
        wandb.log({"train/loss_matching": loss_matching}, step=model.Eiters)
        wandb.log({"train/likelihoodLoss": likelihoodLoss}, step=model.Eiters)
        wandb.log({"train/likelihoodLoss_vid": likelihoodLoss_vid}, step=model.Eiters)
        wandb.log({"train/likelihoodLoss_text": likelihoodLoss_text}, step=model.Eiters)
        if opt.unlikelihood:
            wandb.log({"train/unlikelihoodLoss": unlikelihoodLoss}, step=model.Eiters)
            wandb.log({"train/unlikelihoodLoss_vid": unlikelihoodLoss_vid}, step=model.Eiters)
            wandb.log({"train/unlikelihoodLoss_text": unlikelihoodLoss_text}, step=model.Eiters)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix='', best_epoch=None):
    """save checkpoint at specific path"""
    torch.save(state, prefix + filename)
    # if is_best:
    #     shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')
    if best_epoch is not None:
        os.remove(prefix + 'checkpoint_epoch_%s.pth.tar'%best_epoch)

def decay_learning_rate(opt, optimizer, decay):
    """decay learning rate to the last LR"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr']*decay

def get_learning_rate(optimizer):
    """Return learning rate"""
    lr_list = []
    for param_group in optimizer.param_groups:
        lr_list.append(param_group['lr'])
    return lr_list



if __name__ == '__main__':
    main()
