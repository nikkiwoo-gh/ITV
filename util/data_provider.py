import os
import numpy as np
import torch
import torch.utils.data as data
from util.util import Progbar,getVideoId,clean_str

VIDEO_MAX_LEN=64

class Dataset4ITV(data.Dataset):
    """
    Load captions and video frame features by pre-trained CNN model.
    """

    def __init__(self, cap_file, visual_feat,motion_feat, bow2vec, concept2vec,vocab, n_caption=None, video2frames=None,concept_video_level=None):
        ##specific for tgif collection
        videoid2idxfile=os.path.join('tgif.caption2Videoname.txt')
        with open(videoid2idxfile,'r') as reader:
            lines=reader.readlines()
        tgif_video2idx={}
        for line in lines:
            video_id = line.split(';;')[0].split('#')[0]
            video_idx=line.split(';;')[-1].strip().split('/')[-1]
            video_idx =video_idx[0:-4]
            tgif_video2idx[video_id]=video_idx

        self.tgif_video2idx = tgif_video2idx
        self.concept_video_level = concept_video_level

        ##obtain valid video ids whose concepts are larger than a threshold. Valid videos are to avoid NaN value in the decoding task.
        filelist = os.listdir(self.concept_video_level)
        valid_videoids=[]
        for file in filelist:
            fileid = file.split('.')[0]
            valid_videoids.append(fileid)
        # Captions
        self.captions = {}
        self.concept_all = {}
        self.cap_ids = []
        self.video_ids = set()
        self.concepts = {}
        self.video2frames = video2frames
        self.collections = []
        self.visual_featdirs = {}
        self.motion_featdirs = {}
        cap_file_name = cap_file.split('/')[-1]
        targetCollection = cap_file_name[:cap_file_name.index('.caption')]
        with open(cap_file, 'r', encoding='iso-8859-1') as cap_reader:
            lines = cap_reader.readlines()
        splitor = ':'
        if lines[0].find('::') > 0:
            splitor = '::'
        if lines[0].find(splitor) <0:
            splitor = ' '
        probar = Progbar(len(lines))
        del lines
        with open(cap_file, 'r', encoding='iso-8859-1') as cap_reader:
            for inum,line in enumerate(cap_reader.readlines()):
                probar.add(1)
                cap_id,caption = line.strip().split(splitor,1)
                ori_video_id = getVideoId(cap_id)
                if not ori_video_id in valid_videoids:
                    continue
                motion_video_id = ori_video_id
                visual_video_id = ori_video_id
                if cap_id.find('VATEX') >= 0 :
                    # video_id = 'VATEX_'+video_id.split('_')[1]
                    video_id = ori_video_id.split('_')[1]
                    collection = 'VATEX'
                    motion_video_id = video_id
                    visual_video_id = video_id
                elif cap_id.find('tgif') >=0:
                    collection = 'tgif'
                    motion_video_id = ori_video_id
                    if ori_video_id in tgif_video2idx:
                        visual_video_id = tgif_video2idx[video_id]
                        if not os.path.exists(
                                os.path.join(motion_feat.datadir.replace(targetCollection, collection), 'npy',
                                             motion_video_id + '.npy')):
                            motion_video_id = self.tgif_video2idx[ori_video_id]
                    else:
                        continue
                elif cap_id.find('tv2016train') >= 0:
                    collection = 'tv2016train'
                elif cap_id.find('enc#') >= 0:
                    collection = 'msvd'
                else:
                    collection = 'msrvtt10k'

                self.collections.append(collection)
                visual_featdir = visual_feat.datadir.replace(targetCollection,collection)
                self.visual_featdirs[cap_id] = visual_featdir
                motion_featdir = motion_feat.datadir.replace(targetCollection,collection)
                self.motion_featdirs[cap_id] = motion_featdir

                if visual_video_id not in self.video2frames.keys():
                    print('%s is not have visual feature'%ori_video_id)
                    continue
                self.captions[cap_id] = caption

                self.cap_ids.append(cap_id)
                self.video_ids.add(ori_video_id)

        prob = Progbar(len(self.video_ids))
        for video_id in self.video_ids:
            prob.add(1)
            file = os.path.join(self.concept_video_level, video_id + '.txt')
            with open(file, 'r',encoding='utf-8') as reader:
                conceptline = reader.read()
            self.concept_all[video_id] = conceptline.split(',')

        self.visual_feat = visual_feat
        self.motion_feat = motion_feat
        self.bow2vec = bow2vec
        self.concept2vec = concept2vec
        self.vocab = vocab
        self.length = len(self.cap_ids)
        self.video_ids = list(self.video_ids)
        if n_caption is not None:
            assert len(self.video_ids) * n_caption == self.length, "%d != %d" % (
            len(self.video_ids) * n_caption, self.length)

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]
        video_id = getVideoId(cap_id)
        video_id_num=video_id
        visual_featdir = self.visual_featdirs[cap_id]
        motion_featdir = self.motion_featdirs[cap_id]
        collection = self.collections[index]

        motion_video_id = video_id_num
        visual_video_id = video_id_num
        if collection.find('VATEX')>=0:
            video_id_num = video_id.split('_')[1]
            motion_video_id = video_id_num
            visual_video_id = video_id_num
        elif collection.find('tgif') >= 0:
            visual_video_id = self.tgif_video2idx[video_id_num]
            if not os.path.exists(os.path.join(motion_featdir,'npy',motion_video_id+'.npy')):
                motion_video_id = self.tgif_video2idx[video_id_num]



        # video
        frame_list = self.video2frames[visual_video_id]
        frame_vecs = []
        for frame_id in frame_list:
            featdir=os.path.join(visual_featdir,'npy',frame_id+'.npy')
            try:
                feat = list(np.load(featdir))
                frame_vecs.append(feat)
            except:
                print('cannot load '+featdir)
                continue
        if len(frame_vecs)==0:
            frame_vecs.append(np.zeros(self.visual_feat.ndims))
        frames_tensor = torch.Tensor(frame_vecs)


        try:
            motion_ori = list(np.load(os.path.join(motion_featdir,'npy',motion_video_id+'.npy')))
        except:
            motion_ori = list(np.zeros(self.motion_feat.ndims))
            print('cannot load ' + os.path.join(motion_featdir,'npy',motion_video_id+'.npy'))

        motion_tensor = torch.Tensor(motion_ori)
        # text
        caption_ori = self.captions[cap_id]
        concept_all = self.concept_all[video_id]
        if self.bow2vec is not None:
            cap_bow = self.bow2vec.mapping(caption_ori)
            if cap_bow is None:
                cap_bow = torch.zeros(self.bow2vec.ndims)
            else:
                cap_bow = torch.Tensor(cap_bow)
        else:
            cap_bow = None

        if self.concept2vec is not None:
            cap_concept = self.concept2vec.mapping_exist_concept(','.join(concept_all))
            if cap_concept is None:
                cap_concept = torch.zeros(self.concept2vec.ndims)
            else:
                cap_concept = torch.Tensor(cap_concept)
        else:
            cap_concept = None

        if self.vocab is not None:
            tokens = clean_str(caption_ori)
            caption = []
            caption.append(self.vocab('<start>'))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab('<end>'))
            cap_tensor = torch.Tensor(caption)
        else:
            cap_tensor = None

        return frames_tensor, motion_tensor,cap_tensor, cap_bow,cap_concept,','.join(concept_all), index, cap_id, video_id

    def __len__(self):
        return self.length

class VisDataSet4ITV(data.Dataset):
    """
    Load video frame features by pre-trained CNN model.
    """
    def __init__(self, visual_feat,motion_feat=None, video2frames=None,requred_videolist=None):
        self.visual_feat = visual_feat
        self.visual_feat_path = os.path.join(self.visual_feat.datadir,'npy')
        self.motion_feat = None
        if motion_feat is not None:
            self.motion_feat = motion_feat
            self.motion_feat_path = os.path.join(self.motion_feat.datadir,'npy')
        self.video2frames = video2frames
        if requred_videolist is not None:
            self.video_ids=requred_videolist
        else:
            self.video_ids = list(video2frames.keys())

        self.length = len(self.video_ids)

    def __getitem__(self, index):
        video_id = self.video_ids[index]
        frame_list = self.video2frames[video_id]

        frame_vecs = []
        visual_feat_path = self.visual_feat_path
        motion_feat_path = self.motion_feat_path

        for frame_id in frame_list:
            #print(frame_id)
            try:
                feat = list(np.load(os.path.join(visual_feat_path, frame_id + '.npy')))
            except:
                print('error in loadding '+os.path.join(visual_feat_path, frame_id + '.npy'))
                feat = np.zeros(self.visual_feat.ndims)
            frame_vecs.append(feat)

        frames_tensor = torch.Tensor(frame_vecs)

        if self.motion_feat is not None:
            try:
                motion_ori = list(np.load(os.path.join(motion_feat_path, video_id + '.npy')))
            except:
                print('error in loadding '+os.path.join(motion_feat_path, video_id + '.npy'))

                motion_ori = np.zeros(self.motion_feat.ndims)
            motion_tensor = torch.Tensor(motion_ori)
            return frames_tensor,motion_tensor, index, video_id
        else:
            return frames_tensor, index, video_id

    def __len__(self):
        return self.length

class TxtDataSet4ITV(data.Dataset):
    """
    Load captions
    """
    def __init__(self, cap_file, bow2vec, vocab):
        # Captions
        self.captions = {}
        self.cap_ids = []
        with open(cap_file, 'r', encoding='iso-8859-1') as cap_reader:
        # with open(cap_file, 'r') as cap_reader:
            for line in cap_reader.readlines():
                if len(line.strip().split(' ', 1))<2:
                    continue
                cap_id, caption = line.strip().split(' ', 1)
                if caption == '---------':
                    continue
                self.captions[cap_id] = caption
                self.cap_ids.append(cap_id)
        self.bow2vec = bow2vec
        self.vocab = vocab
        self.length = len(self.cap_ids)

    def __getitem__(self, index):
        cap_id = self.cap_ids[index]

        caption = self.captions[cap_id]
        if self.bow2vec is not None:
            cap_bow = self.bow2vec.mapping(caption)
            if cap_bow is None:
                cap_bow = torch.zeros(self.bow2vec.ndims)
            else:
                cap_bow = torch.Tensor(cap_bow)
        else:
            cap_bow = None

        if self.vocab is not None:
            tokens = clean_str(caption)
            caption = []
            caption.append(self.vocab('<start>'))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab('<end>'))
            cap_tensor = torch.Tensor(caption)
        else:
            cap_tensor = None

        return cap_tensor, cap_bow, index, cap_id

    def __len__(self):
        return self.length

def collate_ITV_vid_txt(data):
    """
    Build mini-batch tensors from a list of (video, caption) tuples.
    """
    # Sort a data list by caption length
    if data[0][2] is not None:
        data.sort(key=lambda x: len(x[2]), reverse=True)
    videos, motions,captions, cap_bows,concept_bows, caption_ori,idxs, cap_ids, video_ids = zip(*data)

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    video_lengths = [min(VIDEO_MAX_LEN, len(frame)) for frame in videos]
    frame_vec_len = len(videos[0][0])
    vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
    videos_origin = torch.zeros(len(videos), frame_vec_len)
    vidoes_mask = torch.zeros(len(videos), max(video_lengths))
    motions = torch.stack(motions, 0) if motions[0] is not None else None
    for i, frames in enumerate(videos):
        end = video_lengths[i]
        vidoes[i, :end, :] = frames[:end, :]
        videos_origin[i, :] = torch.mean(frames, 0)
        vidoes_mask[i, :end] = 1.0

    if captions[0] is not None:
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        target = torch.zeros(len(captions), max(lengths)).long()
        words_mask = torch.zeros(len(captions), max(lengths))
        for i, cap in enumerate(captions):
            end = lengths[i]
            target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
    else:
        target = None
        lengths = None
        words_mask = None

    cap_bows = torch.stack(cap_bows, 0) if cap_bows[0] is not None else None

    concept_bows = torch.stack(concept_bows, 0) if concept_bows[0] is not None else None

    video_data = (vidoes,motions, videos_origin, video_lengths, vidoes_mask)
    text_data = (target, cap_bows, lengths, words_mask)

    return video_data, text_data,concept_bows, caption_ori,idxs, cap_ids, video_ids

def collate_ITV_vid(data):

    videos, motions,idxs, video_ids = zip(*data)

    # Merge videos (convert tuple of 1D tensor to 4D tensor)
    video_lengths = [min(VIDEO_MAX_LEN,len(frame)) for frame in videos]
    frame_vec_len = len(videos[0][0])
    vidoes = torch.zeros(len(videos), max(video_lengths), frame_vec_len)
    videos_origin = torch.zeros(len(videos), frame_vec_len)
    vidoes_mask = torch.zeros(len(videos), max(video_lengths))
    for i, frames in enumerate(videos):
            end = video_lengths[i]
            vidoes[i, :end, :] = frames[:end,:]
            videos_origin[i,:] = torch.mean(frames,0)
            vidoes_mask[i,:end] = 1.0

    motions = torch.stack(motions, 0) if motions[0] is not None else None

    video_data = (vidoes,motions, videos_origin, video_lengths, vidoes_mask)

    return video_data, idxs, video_ids

def collate_txt(data):
    if data[0][0] is not None:
        data.sort(key=lambda x: len(x[0]), reverse=True)
    captions, cap_bows, idxs, cap_ids = zip(*data)

    if captions[0] is not None:
        # Merge captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        target = torch.zeros(len(captions), max(lengths)).long()
        words_mask = torch.zeros(len(captions), max(lengths))
        for i, cap in enumerate(captions):
            end = lengths[i]
            target[i, :end] = cap[:end]
            words_mask[i, :end] = 1.0
    else:
        target = None
        lengths = None
        words_mask = None


    cap_bows = torch.stack(cap_bows, 0) if cap_bows[0] is not None else None

    text_data = (target, cap_bows, lengths, words_mask)

    return text_data, idxs, cap_ids

def get_vid_txt_data_loaders(cap_files, visual_feats,motion_feats, vocab, bow2vec,concept2vec, batch_size=128, num_workers=2, n_caption=2, video2frames=None,concept_video_level=None):
    """
    Returns torch.utils.data.DataLoader for train and validation datasets
    """
    dset = {'train': Dataset4ITV(cap_files['train'], visual_feats['train'],motion_feats['train'], bow2vec,concept2vec, vocab, video2frames=video2frames['train'],concept_video_level=concept_video_level['train']),
            'val': Dataset4ITV(cap_files['val'], visual_feats['val'],motion_feats['val'], bow2vec, concept2vec,vocab, n_caption, video2frames=video2frames['val'],concept_video_level=concept_video_level['val']) }

    data_loaders = {x: torch.utils.data.DataLoader(dataset=dset[x],
                                    batch_size=batch_size,
                                    shuffle=(x=='train'),
                                    pin_memory=True,
                                    num_workers=num_workers,
                                    collate_fn=collate_ITV_vid_txt)
                        for x in cap_files }
    return data_loaders

def get_vis_data_loader(vis_feat,motion_feat,batch_size=100, num_workers=2, video2frames=None,requred_videolist=None):
    dset = VisDataSet4ITV(vis_feat,motion_feat=motion_feat,video2frames=video2frames,requred_videolist=requred_videolist)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_ITV_vid)
    return data_loader

def get_txt_data_loader(cap_file, vocab, bow2vec, batch_size=100, num_workers=2):
    dset = TxtDataSet4ITV(cap_file, bow2vec, vocab)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=collate_txt)
    return data_loader
