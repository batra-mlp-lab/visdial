import argparse
import glob
import h5py
import json
import os
import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-download', action='store_true', help='Whether to download VisDial data')
parser.add_argument('-version', default='1.0', choices=['0.9', '1.0'], help='Version of VisDial to be downloaded')
parser.add_argument('-train_split', default='train', help='Choose the data split: train | trainval', choices=['train', 'trainval'])

# Input files
parser.add_argument('-input_json_train', default='visdial_1.0_train.json', help='Input `train` json file')
parser.add_argument('-input_json_val', default='visdial_1.0_val.json', help='Input `val` json file')
parser.add_argument('-input_json_test', default='visdial_1.0_test.json', help='Input `test` json file')
parser.add_argument('-image_root', default='/path/to/images', help='Path to coco and VisDial val/test images')
parser.add_argument('-input_vocab', default=False, help='Optional vocab file; similar to visdial_params.json')

# Output files
parser.add_argument('-output_json', default='visdial_params.json', help='Output json file')
parser.add_argument('-output_h5', default='visdial_data.h5', help='Output hdf5 file')

# Options
parser.add_argument('-max_ques_len', default=20, type=int, help='Max length of questions')
parser.add_argument('-max_ans_len', default=20, type=int, help='Max length of answers')
parser.add_argument('-max_cap_len', default=40, type=int, help='Max length of captions')
parser.add_argument('-word_count_threshold', default=5, type=int, help='Min threshold of word count to include in vocabulary')


def tokenize_data(data, word_count=False):
    """Tokenize captions, questions and answers, maintain word count
    if required.
    """
    word_counts = {}
    dialogs = data['data']['dialogs']
    # dialogs is a nested dict so won't be copied, just a reference

    print("[%s] Tokenizing captions..." % data['split'])
    for i, dialog in enumerate(tqdm(dialogs)):
        caption = word_tokenize(dialog['caption'])
        dialogs[i]['caption_tokens'] = caption

    print("[%s] Tokenizing questions and answers..." % data['split'])
    q_tokens, a_tokens = [], []
    for q in tqdm(data['data']['questions']):
        q_tokens.append(word_tokenize(q + '?'))

    for a in tqdm(data['data']['answers']):
        a_tokens.append(word_tokenize(a))
    data['data']['question_tokens'] = q_tokens
    data['data']['answer_tokens'] = a_tokens

    print("[%s] Filling missing values in dialog, if any..." % data['split'])
    for i, dialog in enumerate(tqdm(dialogs)):
        # last round of dialog will not have answer for test split
        if 'answer' not in dialog['dialog'][-1]:
            dialog['dialog'][-1]['answer'] = -1
        # right-pad dialog with empty question-answer pairs at the end
        dialog['num_rounds'] = len(dialog['dialog'])
        while len(dialog['dialog']) < 10:
            dialog['dialog'].append({'question': -1, 'answer': -1})
        dialogs[i] = dialog

    if word_count:
        print("[%s] Building word counts from tokens..." % data['split'])
        for i, dialog in enumerate(tqdm(dialogs)):
            caption = dialogs[i]['caption_tokens']
            all_qa = []
            for j in range(10):
                all_qa += q_tokens[dialog['dialog'][j]['question']]
                all_qa += a_tokens[dialog['dialog'][j]['answer']]
            for word in caption + all_qa:
                word_counts[word] = word_counts.get(word, 0) + 1
    print('\n')
    return data, word_counts


def encode_vocab(data, word2ind):
    """Converts string tokens to indices based on given dictionary."""
    dialogs = data['data']['dialogs']
    print("[%s] Encoding caption tokens..." % data['split'])
    for i, dialog in enumerate(tqdm(dialogs)):
        dialogs[i]['caption_tokens'] = [word2ind.get(word, word2ind['UNK']) \
                                        for word in dialog['caption_tokens']]

    print("[%s] Encoding question and answer tokens..." % data['split'])
    q_tokens = data['data']['question_tokens']
    a_tokens = data['data']['answer_tokens']

    for i, q in enumerate(tqdm(q_tokens)):
        q_tokens[i] = [word2ind.get(word, word2ind['UNK']) for word in q]

    for i, a in enumerate(tqdm(a_tokens)):
        a_tokens[i] = [word2ind.get(word, word2ind['UNK']) for word in a]

    data['data']['question_tokens'] = q_tokens
    data['data']['answer_tokens'] = a_tokens
    return data


def create_data_mats(data, params, dtype):
    num_threads = len(data['data']['dialogs'])
    data_mats = {}
    data_mats['img_pos'] = np.arange(num_threads, dtype=np.int)

    print("[%s] Creating caption data matrices..." % data['split'])
    max_cap_len = params.max_cap_len
    captions = np.zeros([num_threads, max_cap_len])
    caption_len = np.zeros(num_threads, dtype=np.int)

    for i, dialog in enumerate(tqdm(data['data']['dialogs'])):
        caption_len[i] = len(dialog['caption_tokens'][0:max_cap_len])
        captions[i][0:caption_len[i]] = dialog['caption_tokens'][0:max_cap_len]
    data_mats['cap_length'] = caption_len
    data_mats['cap'] = captions

    print("[%s] Creating question and answer data matrices..." % data['split'])
    num_rounds = 10
    max_ques_len = params.max_ques_len
    max_ans_len = params.max_ans_len

    ques = np.zeros([num_threads, num_rounds, max_ques_len])
    ans = np.zeros([num_threads, num_rounds, max_ans_len])
    ques_length = np.zeros([num_threads, num_rounds], dtype=np.int)
    ans_length = np.zeros([num_threads, num_rounds], dtype=np.int)

    for i, dialog in enumerate(tqdm(data['data']['dialogs'])):
        for j in range(num_rounds):
            if dialog['dialog'][j]['question'] != -1:
                ques_length[i][j] = len(data['data']['question_tokens'][
                    dialog['dialog'][j]['question']][0:max_ques_len])
                ques[i][j][0:ques_length[i][j]] = data['data']['question_tokens'][
                    dialog['dialog'][j]['question']][0:max_ques_len]
            if dialog['dialog'][j]['answer'] != -1:
                ans_length[i][j] = len(data['data']['answer_tokens'][
                    dialog['dialog'][j]['answer']][0:max_ans_len])
                ans[i][j][0:ans_length[i][j]] = data['data']['answer_tokens'][
                    dialog['dialog'][j]['answer']][0:max_ans_len]

    data_mats['ques'] = ques
    data_mats['ans'] = ans
    data_mats['ques_length'] = ques_length
    data_mats['ans_length'] = ans_length

    print("[%s] Creating options data matrices..." % data['split'])
    # options and answer_index are 1-indexed specifically for lua
    options = np.ones([num_threads, num_rounds, 100])
    num_rounds_list = np.full(num_threads, 10)

    for i, dialog in enumerate(tqdm(data['data']['dialogs'])):
        for j in range(num_rounds):
            num_rounds_list[i] = dialog['num_rounds']
            # v1.0 test does not have options for all dialog rounds
            if 'answer_options' in dialog['dialog'][j]:
                options[i][j] += np.array(dialog['dialog'][j]['answer_options'])

    data_mats['num_rounds'] = num_rounds_list
    data_mats['opt'] = options

    if dtype != 'test':
        print("[%s] Creating ground truth answer data matrices..." % data['split'])
        answer_index = np.zeros([num_threads, num_rounds])
        for i, dialog in enumerate(tqdm(data['data']['dialogs'])):
            for j in range(num_rounds):
                answer_index[i][j] = dialog['dialog'][j]['gt_index'] + 1
        data_mats['ans_index'] = answer_index

    options_len = np.zeros(len(data['data']['answer_tokens']), dtype=np.int)
    options_list = np.zeros([len(data['data']['answer_tokens']), max_ans_len])

    for i, ans_token in enumerate(tqdm(data['data']['answer_tokens'])):
        options_len[i] = len(ans_token[0:max_ans_len])
        options_list[i][0:options_len[i]] = ans_token[0:max_ans_len]

    data_mats['opt_length'] = options_len
    data_mats['opt_list'] = options_list
    return data_mats


def get_image_ids(data, id2path):
    image_ids = [dialog['image_id'] for dialog in data['data']['dialogs']]
    for i, image_id in enumerate(image_ids):
        image_ids[i] = id2path[image_id]
    return image_ids


if __name__ == "__main__":
    args = parser.parse_args()

    if args.download:
        if args.version == '1.0':
            os.system('wget https://www.dropbox.com/s/ix8keeudqrd8hn8/visdial_1.0_train.zip')
            os.system('wget https://www.dropbox.com/s/ibs3a0zhw74zisc/visdial_1.0_val.zip')
        elif args.version == '0.9':
            os.system('wget https://computing.ece.vt.edu/~abhshkdz/data/visdial/visdial_0.9_train.zip')
            os.system('wget https://computing.ece.vt.edu/~abhshkdz/data/visdial/visdial_0.9_val.zip')
        os.system('wget https://www.dropbox.com/s/o7mucbre2zm7i5n/visdial_1.0_test.zip')

        os.system('unzip visdial_%s_train.zip' % args.version)
        os.system('unzip visdial_%s_val.zip' % args.version)
        os.system('unzip visdial_1.0_test.zip')

        args.input_json_train = 'visdial_%s_train.json' % args.version
        args.input_json_val = 'visdial_%s_val.json' % args.version
        args.input_json_test = 'visdial_1.0_test.json'

    print('Reading json...')
    data_train = json.load(open(args.input_json_train, 'r'))
    data_val = json.load(open(args.input_json_val, 'r'))
    data_test = json.load(open(args.input_json_test, 'r'))

    # Tokenizing
    data_train, word_counts_train = tokenize_data(data_train, True)
    data_val, word_counts_val = tokenize_data(data_val, True)
    data_test, _ = tokenize_data(data_test)

    if args.input_vocab == False:
        word_counts_all = dict(word_counts_train)
        # combining the word counts of train and val splits
        if args.train_split == 'trainval':
            for word, count in word_counts_val.items():
                word_counts_all[word] = word_counts_all.get(word, 0) + count

        print('Building vocabulary...')
        word_counts_all['UNK'] = args.word_count_threshold
        vocab = [word for word in word_counts_all \
                if word_counts_all[word] >= args.word_count_threshold]
        print('Words: %d' % len(vocab))
        word2ind = {word: word_ind + 1 for word_ind, word in enumerate(vocab)}
        ind2word = {word_ind: word for word, word_ind in word2ind.items()}
    else:
        print('Loading vocab from %s...' % args.input_vocab)
        vocab_data = json.load(open(args.input_vocab, 'r'))

        word2ind = vocab_data['word2ind']
        for i in word2ind:
            word2ind[i] = int(word2ind[i])

        ind2word = {}
        for i in vocab_data['ind2word']:
            ind2word[int(i)] = vocab_data['ind2word'][i]

    print('Encoding based on vocabulary...')
    data_train = encode_vocab(data_train, word2ind)
    data_val = encode_vocab(data_val, word2ind)
    data_test = encode_vocab(data_test, word2ind)

    print('Creating data matrices...')
    data_mats_train = create_data_mats(data_train, args, 'train')
    data_mats_val = create_data_mats(data_val, args, 'val')
    data_mats_test = create_data_mats(data_test, args, 'test')

    if args.train_split == 'trainval':
        data_mats_trainval = {}
        for key in data_mats_train:
            data_mats_trainval[key] = np.concatenate((data_mats_train[key],
                                                      data_mats_val[key]), axis = 0)

    print('Saving hdf5 to %s...' % args.output_h5)
    f = h5py.File(args.output_h5, 'w')
    if args.train_split == 'train':
        for key in data_mats_train:
            f.create_dataset(key + '_train', dtype='uint32', data=data_mats_train[key])

        for key in data_mats_val:
            f.create_dataset(key + '_val', dtype='uint32', data=data_mats_val[key])

    elif args.train_split == 'trainval':
        for key in data_mats_trainval:
            f.create_dataset(key + '_train', dtype='uint32', data=data_mats_trainval[key])

    for key in data_mats_test:
        f.create_dataset(key + '_test', dtype='uint32', data=data_mats_test[key])
    f.close()

    out = {}
    out['ind2word'] = ind2word
    out['word2ind'] = word2ind

    print('Preparing image paths with image_ids...')
    id2path = {}
    # NOTE: based on assumption that image_id is unique across all splits
    for image_path in tqdm(glob.iglob(os.path.join(args.image_root, '*', '*.jpg'))):
        id2path[int(image_path[-12:-4])] = '/'.join(image_path.split('/')[-2:])

    out['unique_img_train'] = get_image_ids(data_train, id2path)
    out['unique_img_val'] = get_image_ids(data_val, id2path)
    out['unique_img_test'] = get_image_ids(data_test, id2path)
    if args.train_split == 'trainval':
        out['unique_img_train'] += out['unique_img_val']
        out.pop('unique_img_val')
    print('Saving json to %s...' % args.output_json)
    json.dump(out, open(args.output_json, 'w'))
