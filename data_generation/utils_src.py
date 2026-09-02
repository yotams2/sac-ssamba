import os
import random
import numpy as np
import scipy
import scipy.io
import scipy.signal
import soundfile
import webrtcvad
from torch.utils.data import Dataset

def explore_corpus(path, file_extension):
        directory_tree = {}
        path_set = []
        for item in os.listdir(path):   
            if os.path.isdir( os.path.join(path, item) ):
                directory_tree[item], path_set_temp = explore_corpus( os.path.join(path, item), file_extension )
                path_set += path_set_temp
            elif item.split(".")[-1] == file_extension:
                directory_tree[ item.split(".")[0] ] = os.path.join(path, item)
                path_set += [os.path.join(path, item)]
        return directory_tree, path_set

def pad_cut_sig_sameutt(sig, nsample_desired):
    """ Pad (by repeating the same utterance) and cut signal to desired length
        Args:       sig             - signal (nsample, )
                    nsample_desired - desired sample length
        Returns:    sig_pad_cut     - padded and cutted signal (nsample_desired,)
    """ 
    nsample = sig.shape[0]
    while nsample < nsample_desired:
        sig = np.concatenate((sig, sig), axis=0)
        nsample = sig.shape[0]
    st = np.random.randint(0, nsample - nsample_desired+1)
    ed = st + nsample_desired
    sig_pad_cut = sig[st:ed]

    return sig_pad_cut


def pad_cut_sig_samespk(utt_path_list, current_utt_idx, nsample_desired, fs_desired):
    """ Pad (by adding utterance of the same spearker) and cut signal to desired length
        Args:       utt_path_list             - 
                    current_utt_idx
                    nsample_desired - desired sample length
                    fs_desired
        Returns:    sig_pad_cut     - padded and cutted signal (nsample_desired,)
    """ 
    sig = np.array([])
    nsample = sig.shape[0]
    while nsample < nsample_desired:
        utterance, fs = soundfile.read(utt_path_list[current_utt_idx])
        if fs != fs_desired:
            utterance = scipy.signal.resample_poly(utterance, up=fs_desired, down=fs)
            raise Warning(f'Signal is downsampled from {fs} to {fs_desired}')
        sig = np.concatenate((sig, utterance), axis=0)
        nsample = sig.shape[0]
        current_utt_idx += 1
        if current_utt_idx >= len(utt_path_list): current_utt_idx=0
    st = np.random.randint(0, nsample - nsample_desired+1)
    ed = st + nsample_desired
    sig_pad_cut = sig[st:ed]

    return sig_pad_cut

class WSJ0Dataset(Dataset):
    """ WSJ0Dataset (after 20240403)
        train: /tr 81h (both speaker independent and dependent)
        val: /dt 5h
        test: /et 5h
        spk/wav
    """
    def __init__(self, path, T, fs, num_source=1, size=None, exclude_spk_ids=None):

        self.corpus, self.paths = explore_corpus(path, 'wav')
        
        # Filter excluded speakers
        if exclude_spk_ids is not None:
             for spk_id in exclude_spk_ids:
                 if spk_id in self.corpus:
                     del self.corpus[spk_id]

        # self.corpus is expected to map speaker_id -> {utt_id: utt_path, ...}
        # Keep one entry per speaker: spkWAVs will be a list of dicts (utt_id->path)
        # and spkIDs will be the matching list of speaker IDs
        self.spkWAVs = []
        self.spkIDs = []
        for spk_id, spks in self.corpus.items():
            # spks should be a dict mapping utterance id -> wav path
            self.spkWAVs.append(spks)
            self.spkIDs.append(spk_id)

        # self.paths.sort()
        # random.shuffle(self.paths)
        self.fs = fs
        self.T = T
        self.sum_source = num_source
        self.sz = len(self.spkIDs) if size is None else size 

    def __len__(self):
        return self.sz

    def __getitem__(self, idx):
        if idx < 0: idx = len(self.spkIDs) + idx
        elif idx >= len(self.spkIDs): idx = idx % len(self.spkIDs)

        # random speaker IDs
        spkID_list = [self.spkIDs[idx]]
        idx_list = [idx]
        while(len(set(spkID_list))<self.sum_source):
            idx_othersources = np.random.randint(0, len(self.spkIDs))
            spkID_list += [self.spkIDs[idx_othersources]]
            idx_list += [idx_othersources]

        # read speech signals
        s_shape_desired = int(self.T * self.fs)
        s_sources = []
        for source_idx in range(self.sum_source):
            spkID = spkID_list[source_idx]
            spkWAVs = self.spkWAVs[idx_list[source_idx]]
            utt_paths = list(spkWAVs.values())
            # Get a random speech utterance from specific speaker
            utt_idx = np.random.randint(0, len(utt_paths))
            # read from the selected speaker's utterance paths (utt_paths),
            # not from the global self.paths (that would be the wrong list)
            s, fs = soundfile.read(utt_paths[utt_idx], dtype='float32')
            if fs != self.fs:
                s = scipy.signal.resample_poly(s, up=self.fs, down=fs)
                raise Warning('WSJ0 is downsampled to requrired frequency~')
            s = pad_cut_sig_samespk(utt_paths, utt_idx, s_shape_desired, self.fs) # pad by the same spk
            s -= s.mean()

            s_sources += [s]
        s_sources = np.array(s_sources).transpose(1,0)

        return s_sources #, np.ones_like(s_sources)
    

class LibriSpeechDataset(Dataset):
    """ LibriSpeechDataset (about 1000h)
        https://www.openslr.org/12
        spk/chapter/spk-chapter-utterance.flac
    """

    def _cleanSilences(self, s, aggressiveness, return_vad=False):
        if not hasattr(self, 'vad') or self.vad is None:
            self.vad = webrtcvad.Vad()

        self.vad.set_mode(aggressiveness)

        vad_out = np.zeros_like(s)
        vad_frame_len = int(10e-3 * self.fs)  # 0.001s,16samples gives one same vad results
        n_vad_frames = len(s) // vad_frame_len # 1/0.001s
        for frame_idx in range(n_vad_frames):
            frame = s[frame_idx * vad_frame_len: (frame_idx + 1) * vad_frame_len]
            frame_bytes = (frame * 32767).astype('int16').tobytes()
            vad_out[frame_idx*vad_frame_len: (frame_idx+1)*vad_frame_len] = self.vad.is_speech(frame_bytes, self.fs)
        s_clean = s * vad_out

        return (s_clean, vad_out) if return_vad else s_clean

    def __init__(self, path, T, fs, num_source, size=None, return_vad=False, readers_range=None, clean_silence=True):
        self.corpus, _ = explore_corpus(path, 'flac')
        if readers_range is not None:
            for key in list(map(int, self.nChapters.keys())):
                if int(key) < readers_range[0] or int(key) > readers_range[1]:
                    del self.corpus[key]

        self.nReaders = len(self.corpus)
        self.nChapters = {reader: len(self.corpus[reader]) for reader in self.corpus.keys()}
        self.nUtterances = {reader: {
        chapter: len(self.corpus[reader][chapter]) for chapter in self.corpus[reader].keys()
        } for reader in self.corpus.keys()}

        self.chapterList = []
        for chapters in list(self.corpus.values()):
            self.chapterList += list(chapters.values())
        # self.chapterList.sort()

        self.fs = fs
        self.T = T
        self.num_source = num_source

        self.clean_silence = clean_silence
        self.return_vad = return_vad

        self.sz = len(self.chapterList) if size is None else size

    def __len__(self):
        return self.sz

    def __getitem__(self, idx):
        if idx < 0: idx = len(self) + idx
        while idx >= len(self.chapterList): idx -= len(self.chapterList)

        s_sources = []
        s_clean_sources = []
        vad_out_sources = []
        spkID_list = []

        for source_idx in range(self.num_source):
            if source_idx==0:
                chapter = self.chapterList[idx]
                utts = list(chapter.keys())
                spkID = utts[0].split('-')[0]
                spkID_list += [spkID]
            else:
                while(len(set(spkID_list))<=source_idx):
                    idx_othersources = np.random.randint(0, len(self.chapterList))
                    chapter = self.chapterList[idx_othersources]
                    utts = list(chapter.keys())
                    spkID = utts[0].split('-')[0]
                    spkID_list += [spkID]

            utt_paths = list(chapter.values())
            s_shape_desired = int(self.T * self.fs)
            s_clean = np.zeros((s_shape_desired, 1)) # random initialization
            while np.sum(s_clean) == 0: # avoid full-zero s_clean
                # Get a random speech segment from the selected chapter
                utt_idx = np.random.randint(0, len(chapter))
                s = pad_cut_sig_samespk(utt_paths, utt_idx, s_shape_desired, self.fs) # pad by the same spk & chapter
                s -= s.mean()

                # Clean silences, it starts with the highest aggressiveness of webrtcvad,
                # but it reduces it if it removes more than the 66% of the samples
                s_clean, vad_out = self._cleanSilences(s, 3, return_vad=True)
                if np.count_nonzero(s_clean) < len(s_clean) * 0.66:
                    s_clean, vad_out = self._cleanSilences(s, 2, return_vad=True)
                if np.count_nonzero(s_clean) < len(s_clean) * 0.66:
                    s_clean, vad_out = self._cleanSilences(s, 1, return_vad=True)

            s_sources += [s]
            s_clean_sources += [s_clean]
            vad_out_sources += [vad_out]

        s_sources = np.array(s_sources).transpose(1,0)
        s_clean_sources = np.array(s_clean_sources).transpose(1,0)
        vad_out_sources = np.array(vad_out_sources).transpose(1,0)

        # scipy.io.savemat('source_data.mat',{'s':s_sources, 's_clean':s_clean_sources})


        if self.clean_silence:
            return (s_clean_sources, vad_out_sources) if self.return_vad else s_clean_sources
        else:
            return (s_sources, vad_out_sources) if self.return_vad else s_sources


class LibriSpeechForSimuDataset(Dataset):
    """ LibriSpeech dataset for the simulation / spatialization pipeline.

        Matches the WSJ0Dataset interface expected by gen_simu.py:
          - Indexed by speaker (idx → speaker)
          - __getitem__ returns np.ndarray of shape (nsample, num_source)

        Segment extraction (Option C):
          1. Pick a random chapter from the selected speaker
          2. Concatenate utterances from that chapter via pad_cut_sig_samespk
          3. Apply WebRTC VAD silence cleaning (zero out non-speech frames)
          4. Retry with reduced aggressiveness or different chapter if >66% is silence

        Args:
            path:            Root directory of a LibriSpeech split (e.g. train-clean-100)
            T:               Desired segment duration in seconds
            fs:              Sample rate (Hz)
            num_source:      Number of simultaneous sources (speakers) per sample
            size:            Optional fixed dataset size; defaults to number of speakers
            include_spk_ids: If provided, only include these speaker IDs (set or list).
                             Used to split dev-clean into preval / pretest.
    """

    def __init__(self, path, T, fs, num_source=1, size=None, include_spk_ids=None):
        self.corpus, self.paths = explore_corpus(path, 'flac')

        # Optionally restrict to a subset of speakers (for dev-clean splitting)
        if include_spk_ids is not None:
            include_set = set(include_spk_ids)
            self.corpus = {k: v for k, v in self.corpus.items() if k in include_set}

        # Build per-speaker, per-chapter utterance path lists.
        # LibriSpeech structure: spk/chapter/spk-chapter-utt.flac
        # explore_corpus returns: {spk: {chapter: {utt_id: path, ...}, ...}, ...}
        self.spk_chapters = {}   # spk_id -> list of [chapter_utt_paths_list, ...]
        self.spkIDs = []

        for spk_id, chapters in self.corpus.items():
            chapter_lists = []
            if isinstance(chapters, dict):
                for chapter_id, utts in sorted(chapters.items()):
                    if isinstance(utts, dict):
                        chapter_lists.append(sorted(utts.values()))
                    elif isinstance(utts, str):
                        # Single file directly under speaker (unlikely for LibriSpeech)
                        chapter_lists.append([utts])
            if chapter_lists:
                self.spk_chapters[spk_id] = chapter_lists
                self.spkIDs.append(spk_id)

        self.spkIDs.sort()  # deterministic ordering

        self.fs = fs
        self.T = T
        self.num_source = num_source
        self.sz = len(self.spkIDs) if size is None else size
        self.vad = None  # lazily initialized (must be None for pickling in mp.Pool)

    def __len__(self):
        return self.sz

    def _clean_silences(self, s, aggressiveness):
        """ Zero out non-speech frames using WebRTC VAD.
            Returns (cleaned_signal, vad_mask).
        """
        if self.vad is None:
            self.vad = webrtcvad.Vad()
        self.vad.set_mode(aggressiveness)

        vad_out = np.zeros_like(s)
        vad_frame_len = int(10e-3 * self.fs)  # 10 ms frames
        n_vad_frames = len(s) // vad_frame_len
        for frame_idx in range(n_vad_frames):
            frame = s[frame_idx * vad_frame_len: (frame_idx + 1) * vad_frame_len]
            frame_bytes = (frame * 32767).astype('int16').tobytes()
            vad_out[frame_idx * vad_frame_len: (frame_idx + 1) * vad_frame_len] = \
                self.vad.is_speech(frame_bytes, self.fs)
        s_clean = s * vad_out
        return s_clean, vad_out

    def __getitem__(self, idx):
        if idx < 0: idx = len(self.spkIDs) + idx
        elif idx >= len(self.spkIDs): idx = idx % len(self.spkIDs)

        # Select distinct speakers for multi-source scenarios
        spkID_list = [self.spkIDs[idx]]
        idx_list = [idx]
        while len(set(spkID_list)) < self.num_source:
            idx_other = np.random.randint(0, len(self.spkIDs))
            spkID_list.append(self.spkIDs[idx_other])
            idx_list.append(idx_other)

        s_shape_desired = int(self.T * self.fs)
        s_sources = []

        for source_idx in range(self.num_source):
            spk_id = spkID_list[source_idx]
            chapters = self.spk_chapters[spk_id]

            # Try up to max_retries to get a segment with sufficient speech
            s_clean = np.zeros(s_shape_desired)
            max_retries = 5
            for _ in range(max_retries):
                # Pick a random chapter from this speaker
                chapter_idx = np.random.randint(0, len(chapters))
                utt_paths = chapters[chapter_idx]
                utt_idx = np.random.randint(0, len(utt_paths))

                # Concatenate same-chapter utterances and extract a T-second window
                s = pad_cut_sig_samespk(utt_paths, utt_idx, s_shape_desired, self.fs)
                s -= s.mean()

                # VAD silence cleaning - start aggressive, relax if too much is removed
                s_clean, _ = self._clean_silences(s, 3)
                if np.count_nonzero(s_clean) < len(s_clean) * 0.66:
                    s_clean, _ = self._clean_silences(s, 2)
                if np.count_nonzero(s_clean) < len(s_clean) * 0.66:
                    s_clean, _ = self._clean_silences(s, 1)

                if np.sum(s_clean ** 2) > 1e-10:
                    break  # valid segment found

            s_sources.append(s_clean)

        s_sources = np.array(s_sources).transpose(1, 0)
        return s_sources


class VoxCeleb1ForSimuDataset(Dataset):
    """ VoxCeleb1 dataset for room simulation / spatialization.
        Uses same-speaker utterance concatenation (pad_cut_sig_samespk) for short clips < T=8.0s,
        and dry signal pre-convolution cropping for clips > T=8.0s.
        Preserves official verification/identification splits from veri_test_class.txt.
    """
    def __init__(self, vox1_root, meta_file, split='pretrain', T=8.0, fs=16000, num_source=1, size=None):
        self.vox1_root = Path(vox1_root)
        self.T = T
        self.fs = fs
        self.num_source = num_source
        self.split = split

        split_map = {'pretrain': 1, 'preval': 2, 'pretest': 3}
        target_split_id = split_map.get(split, 1)

        # Parse veri_test_class.txt / iden_split.txt
        # Lines: <split_id> <rel_path>  (e.g., "1 id10001/1zcIwhmdeo4/00001.wav")
        self.items = [] # list of (rel_path, full_path, spk_id, label_idx)
        self.spk2utts = {} # spk_id -> list of full_paths

        with open(meta_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 2:
                    continue
                split_id = int(parts[0])
                rel_path = parts[1] # "id10001/1zcIwhmdeo4/00001.wav"
                spk_id = rel_path.split('/')[0] # "id10001"
                label_idx = int(spk_id[2:]) - 10001 # 0..1250

                full_path = str(self.vox1_root / 'wav' / rel_path)

                if spk_id not in self.spk2utts:
                    self.spk2utts[spk_id] = []
                self.spk2utts[spk_id].append(full_path)

                if split_id == target_split_id:
                    self.items.append((rel_path, full_path, spk_id, label_idx))

        self.sz = len(self.items) if size is None else min(size, len(self.items))

    def __len__(self):
        return self.sz

    def __getitem__(self, idx):
        if idx < 0: idx = len(self.items) + idx
        elif idx >= len(self.items): idx = idx % len(self.items)

        rel_path, full_path, spk_id, label_idx = self.items[idx]
        spk_utts = self.spk2utts[spk_id]
        current_utt_idx = spk_utts.index(full_path) if full_path in spk_utts else 0

        s_shape_desired = int(self.T * self.fs)

        # Concatenate same-speaker dry signal to T=8.0s before RIR convolution
        s = pad_cut_sig_samespk(spk_utts, current_utt_idx, s_shape_desired, self.fs)
        s -= s.mean()

        s_sources = np.expand_dims(s, axis=1) # [nsample, 1]
        metadata = {
            'speaker_id': spk_id,
            'label_idx': label_idx,
            'orig_rel_path': rel_path,
        }
        return s_sources, metadata


class IEMOCAPForSimuDataset(Dataset):
    """ IEMOCAP dataset for room simulation / spatialization.
        Uses exact fold1 evaluation split from s3prl:
          - pretest: All utterances in Session1 (test_meta_data.json)
          - pretrain: 80% of Session2-5 (train_meta_data.json) via torch.manual_seed(0) random_split
          - preval: 20% of Session2-5 (train_meta_data.json) via torch.manual_seed(0) random_split
        T=6.0s container. Labels: neu (0), hap (1, includes exc), ang (2), sad (3).
    """
    def __init__(self, iemocap_root, meta_dir, split='pretrain', test_fold='fold1', T=6.0, fs=16000, size=None):
        import json
        self.iemocap_root = Path(iemocap_root)
        self.T = T
        self.fs = fs

        fold_dir = Path(meta_dir) / test_fold.replace('fold', 'Session')
        test_path = fold_dir / 'test_meta_data.json'
        train_path = fold_dir / 'train_meta_data.json'

        class_dict = {'neu': 0, 'hap': 1, 'ang': 2, 'sad': 3}

        if split == 'pretest':
            with open(test_path, 'r') as f:
                data = json.load(f)['meta_data']
            self.items = data
        else:
            with open(train_path, 'r') as f:
                data = json.load(f)['meta_data']
            
            # Deterministic 80/20 split matching s3prl DownstreamExpert
            n_total = len(data)
            n_train = int(0.8 * n_total)
            indices = list(range(n_total))
            rng = random.Random(0)
            rng.shuffle(indices)
            
            train_indices = indices[:n_train]
            val_indices = indices[n_train:]
            
            if split == 'pretrain':
                self.items = [data[i] for i in train_indices]
            else: # preval
                self.items = [data[i] for i in val_indices]

        self.class_dict = class_dict
        self.sz = len(self.items) if size is None else min(size, len(self.items))

    def __len__(self):
        return self.sz

    def __getitem__(self, idx):
        if idx < 0: idx = len(self.items) + idx
        elif idx >= len(self.items): idx = idx % len(self.items)

        item = self.items[idx]
        rel_path = item['path'] # e.g. "Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav"
        full_path = str(self.iemocap_root / rel_path)

        label_str = item['label']
        label_idx = self.class_dict[label_str]
        speaker_id = item.get('speaker', rel_path.split('/')[-1].split('_')[0])
        session = rel_path.split('/')[0]

        # Read audio
        s, fs = soundfile.read(full_path)
        if fs != self.fs:
            s = scipy.signal.resample_poly(s, up=self.fs, down=fs)

        if s.ndim > 1:
            s = s[:, 0]

        s_shape_desired = int(self.T * self.fs)
        if len(s) < s_shape_desired:
            pad = np.zeros(s_shape_desired - len(s))
            s = np.concatenate([s, pad])
        else:
            s = s[:s_shape_desired]

        s -= s.mean()
        s_sources = np.expand_dims(s, axis=1) # [nsample, 1]

        metadata = {
            'emotion_label': label_str,
            'emotion_idx': label_idx,
            'speaker_id': speaker_id,
            'session': session,
            'orig_rel_path': rel_path,
        }
        return s_sources, metadata


class SpeechCommandsForSimuDataset(Dataset):
    """ Google Speech Commands v2 dataset for room simulation / spatialization.
        Uses T=2.0s container (1.0s dry keyword placed at start, remaining 1.0s zero-padded)
        so RIR convolution produces 1.0s keyword + 1.0s natural reverberant decay tail.
        Splits: pretrain (train_list.txt), preval (validation_list.txt), pretest (testing_list.txt).
    """
    def __init__(self, sc_root, label_csv, split='pretrain', T=2.0, fs=16000, size=None):
        self.sc_root = Path(sc_root)
        self.T = T
        self.fs = fs

        # Load label mapping
        label_set = np.loadtxt(label_csv, delimiter=',', dtype='str')
        self.label_map = {}
        for i in range(1, len(label_set)):
            # label_set[i] is like ['0', 'yes', "'00'"]
            key = eval(label_set[i][2]) if label_set[i][2].startswith("'") else label_set[i][2]
            self.label_map[label_set[i][0]] = key

        split_file_map = {
            'pretrain': 'train_list.txt',
            'preval': 'validation_list.txt',
            'pretest': 'testing_list.txt',
        }
        list_file = self.sc_root / split_file_map.get(split, 'train_list.txt')

        self.items = []
        with open(list_file, 'r') as f:
            for line in f:
                rel_path = line.strip()
                if not rel_path:
                    continue
                keyword = rel_path.split('/')[0]
                if keyword in self.label_map:
                    label_idx = int(self.label_map[keyword])
                    full_path = str(self.sc_root / rel_path)
                    self.items.append((rel_path, full_path, keyword, label_idx))

        self.sz = len(self.items) if size is None else min(size, len(self.items))

    def __len__(self):
        return self.sz

    def __getitem__(self, idx):
        if idx < 0: idx = len(self.items) + idx
        elif idx >= len(self.items): idx = idx % len(self.items)

        rel_path, full_path, keyword, label_idx = self.items[idx]

        s, fs = soundfile.read(full_path)
        if fs != self.fs:
            s = scipy.signal.resample_poly(s, up=self.fs, down=fs)

        if s.ndim > 1:
            s = s[:, 0]

        s_shape_desired = int(self.T * self.fs)
        if len(s) < s_shape_desired:
            pad = np.zeros(s_shape_desired - len(s))
            s = np.concatenate([s, pad])
        else:
            s = s[:s_shape_desired]

        s -= s.mean()
        s_sources = np.expand_dims(s, axis=1) # [nsample, 1]

        metadata = {
            'keyword': keyword,
            'keyword_idx': label_idx,
            'file_id': rel_path.split('/')[-1],
            'orig_rel_path': rel_path,
        }
        return s_sources, metadata