import os
import json
import h5py
import argparse
import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('-download', action='store_true', help='Whether to download VisDial v0.9 data')
parser.add_argument('-train_split', default='train', help='Choose the data split: train | trainval', choices=['train', 'trainval'])

# Input files
parser.add_argument('-input_json_train', default='visdial_0.9_train.json', help='Input `train` json file')
parser.add_argument('-input_json_val', default='visdial_0.9_val.json', help='Input `val` json file')
parser.add_argument('-input_json_test', default='visdial_0.9_test.json', help='Input `test` json file')

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


def create_data_mats(data, args, dtype):
    num_threads = len(data['data']['dialogs'].keys())
    data_mats = {}

    print("[%s] Creating image id and caption data matrices..." % data['split'])
    max_cap_len = params.max_cap_len
    captions = np.zeros([num_threads, max_cap_len])
    caption_len = np.zeros(num_threads, dtype=np.int)
    image_ids = [dialog['image_id'] for dialog in data['data']['dialogs']]
    image_list = []

    for i, dialog in enumerate(tqdm(data['data']['dialogs'])):
        caption_len[i] = len(dialog['caption_tokens'][0:max_cap_len])
        captions[i][0:caption_len[i]] = dialog['caption_tokens'][0:max_cap_len]

    for image_id in tqdm(image_ids):
        path = '%s2014/COCO_%s2014_%012d.jpg'
        if dtype == 'test':
            path = '%s2017/VisualDialog_%s2017_%012d.jpg'
        image_list.append(path % (dtype, dtype, image_id))

    data_mats['cap_length_' + dtype] = caption_len
    data_mats['cap_' + dtype] = captions
    data_mats['img_pos_' + dtype] = np.arange(len(image_ids), dtype=np.int)

    print("[%s] Creating question and answer data matrices..." % data['split'])
    num_rounds = 10
    max_ques_len = params.max_ques_len
    max_ans_len = params.max_ans_len

    q = np.zeros([num_threads, num_rounds, max_ques_len])
    a = np.zeros([num_threads, num_rounds, max_ans_len])
    q_len = np.zeros([num_threads, num_rounds], dtype=np.int)
    a_len = np.zeros([num_threads, num_rounds], dtype=np.int)

    for i, dialog in enumerate(tqdm(data['data']['dialogs'])):
        for j in range(num_rounds):
            if dialog['dialog'][j]['question'] != -1:
                q_len[i][j] = len(data['data']['question_tokens'][
                    dialog['dialog'][j]['question']][0:max_ques_len])
                q[i][j][0:q_len[i][j]] = data['data']['question_tokens'][
                    dialog['dialog'][j]['question']][0:max_ques_len]
            if dialog['dialog'][j]['answer'] != -1:
                a_len[i][j] = len(data['data']['answer_tokens'][
                    dialog['dialog'][j]['answer']][0:max_ans_len])
                a[i][j][0:a_len[i][j]] = data['data']['answer_tokens'][
                    dialog['dialog'][j]['answer']][0:max_ans_len]

    data_mats['ques_' + dtype] = q
    data_mats['ans_' + dtype] = a
    data_mats['ques_length_' + dtype] = q_len
    data_mats['ans_length_' + dtype] = a_len

    print("[%s] Creating options data matrices..." % data['split'])
    if dtype == 'test':
        num_rounds_list = np.full(num_threads, 10)
        options = np.zeros([num_threads, 1, 100])
        for i, dialog in enumerate(tqdm(data['data']['dialogs'])):
            for j in range(num_rounds):
                # options and answer_index are 1-indexed specifically for lua
                num_rounds_list[i] = dialog['num_rounds']
            options[i][0] = np.array(dialog['dialog'][num_rounds_list[i] - 1]['answer_options']) + 1
        data_mats['num_rounds_' + dtype] = num_rounds_list
    else:
        answer_index = np.zeros([num_threads, num_rounds])
        options = np.zeros([num_threads, num_rounds, 100])
        for i, dialog in enumerate(tqdm(data['data']['dialogs'])):
            for j in range(num_rounds):
                answer_index[i][j] = dialog['dialog'][j]['gt_index'] + 1
                options[i][j] = np.array(dialog['dialog'][j]['answer_options']) + 1
        data_mats['ans_index_' + dtype] = num_rounds_list
    data_mats['opt_' + dtype] = options

    options_len = np.zeros(len(data['data']['answer_tokens']), dtype=np.int)
    options_list = np.zeros([len(data['data']['answer_tokens']), max_ans_len])

    for i, a in enumerate(tqdm(data['data']['answer_tokens'])):
        options_len[i] = len(a[0:max_ans_len])
        options_list[i][0:options_len[i]] = a[i][0:max_ans_len]

    data_mats['opt_length_' + dtype] = options_len
    data_mats['opt_list_' + dtype] = options_list
    return data_mats


if __name__ == "__main__":
    args = parser.parse_args()

    if args.download == 1:
        os.system('wget https://computing.ece.vt.edu/~abhshkdz/data/visdial/visdial_0.9_train.zip')
        os.system('wget https://computing.ece.vt.edu/~abhshkdz/data/visdial/visdial_0.9_val.zip')
        # if args.train_split == 'trainval':
            #TODO Add download path to visdial_0.9_test.zip

        os.system('unzip visdial_0.9_train.zip')
        os.system('unzip visdial_0.9_val.zip')
        if args.train_split == 'trainval':
            os.system('unzip visdial_0.9_test.zip')

    print('Reading json...')
    data_train = json.load(open(args.input_json_train, 'r'))
    data_val = json.load(open(args.input_json_val, 'r'))
    if args.train_split == 'trainval':
        data_test = json.load(open(args.input_json_test, 'r'))

    # Tokenizing
    data_train, word_counts_train = tokenize_data(data_train, True)
    if args.train_split == 'train':
        data_val, _ = tokenize_data(data_val)
    elif args.train_split == 'trainval':
        data_val, word_counts_val = tokenize_data(data_val, True)
        data_test, _ = tokenize_data(data_test)

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

    print('Encoding based on vocabulary...')
    data_train = encode_vocab(data_train, word2ind)
    data_val = encode_vocab(data_val, word2ind)
    if args.train_split == 'trainval':
        data_test = encode_vocab(data_test, word2ind)

    print('Creating data matrices...')
    captions_train, captions_train_len, questions_train, questions_train_len, answers_train, answers_train_len, options_train, options_train_list, options_train_len, answers_train_index, images_train_index, images_train_list, _ = create_data_mats(data_train, args, 'train')
    captions_val, captions_val_len, questions_val, questions_val_len, answers_val, answers_val_len, options_val, options_val_list, options_val_len, answers_val_index, images_val_index, images_val_list, _ = create_data_mats(data_val, args, 'val')

    if args.train_split == 'trainval':
        captions_trainval = np.concatenate((captions_train, captions_val), axis = 0)
        captions_trainval_len = np.concatenate((captions_train_len, captions_val_len), axis = 0)
        questions_trainval = np.concatenate((questions_train, questions_val), axis = 0)
        questions_trainval_len = np.concatenate((questions_train_len, questions_val_len), axis = 0)
        answers_trainval = np.concatenate((answers_train, answers_val), axis = 0)
        answers_trainval_len = np.concatenate((answers_train_len, answers_val_len), axis = 0)
        options_trainval = np.concatenate((options_train, options_val + len(ans_train_inds)), axis = 0)
        options_trainval_list = np.concatenate((options_train_list, options_val_list), axis = 0)
        options_trainval_len = np.concatenate((options_train_len, options_val_len), axis = 0)
        answers_trainval_index = np.concatenate((answers_train_index, answers_val_index), axis = 0)
        images_trainval_index = np.concatenate((images_train_index, images_val_index + images_train_index.shape[0]), axis = 0)
        images_trainval_list = images_train_list + images_val_list

        captions_test, captions_test_len, questions_test, questions_test_len, answers_test, answers_test_len, options_test, options_test_list, options_test_len, _, images_test_index, images_test_list, num_rounds_test = create_data_mats(data_test_toks, ques_test_inds, ans_test_inds, args, 'test')

    print('Saving hdf5...')
    f = h5py.File(args.output_h5, 'w')
    if args.train_split == 'train':
        f.create_dataset('ques_train', dtype='uint32', data=questions_train)
        f.create_dataset('ques_length_train', dtype='uint32', data=questions_train_len)
        f.create_dataset('ans_train', dtype='uint32', data=answers_train)
        f.create_dataset('ans_length_train', dtype='uint32', data=answers_train_len)
        f.create_dataset('ans_index_train', dtype='uint32', data=answers_train_index)
        f.create_dataset('cap_train', dtype='uint32', data=captions_train)
        f.create_dataset('cap_length_train', dtype='uint32', data=captions_train_len)
        f.create_dataset('opt_train', dtype='uint32', data=options_train)
        f.create_dataset('opt_length_train', dtype='uint32', data=options_train_len)
        f.create_dataset('opt_list_train', dtype='uint32', data=options_train_list)
        f.create_dataset('img_pos_train', dtype='uint32', data=images_train_index)

        f.create_dataset('ques_val', dtype='uint32', data=questions_val)
        f.create_dataset('ques_length_val', dtype='uint32', data=questions_val_len)
        f.create_dataset('ans_val', dtype='uint32', data=answers_val)
        f.create_dataset('ans_length_val', dtype='uint32', data=answers_val_len)
        f.create_dataset('ans_index_val', dtype='uint32', data=answers_val_index)
        f.create_dataset('cap_val', dtype='uint32', data=captions_val)
        f.create_dataset('cap_length_val', dtype='uint32', data=captions_val_len)
        f.create_dataset('opt_val', dtype='uint32', data=options_val)
        f.create_dataset('opt_length_val', dtype='uint32', data=options_val_len)
        f.create_dataset('opt_list_val', dtype='uint32', data=options_val_list)
        f.create_dataset('img_pos_val', dtype='uint32', data=images_val_index)

    elif args.train_split == 'trainval':
        f.create_dataset('ques_train', dtype='uint32', data=questions_trainval)
        f.create_dataset('ques_length_train', dtype='uint32', data=questions_trainval_len)
        f.create_dataset('ans_train', dtype='uint32', data=answers_trainval)
        f.create_dataset('ans_length_train', dtype='uint32', data=answers_trainval_len)
        f.create_dataset('ans_index_train', dtype='uint32', data=answers_trainval_index)
        f.create_dataset('cap_train', dtype='uint32', data=captions_trainval)
        f.create_dataset('cap_length_train', dtype='uint32', data=captions_trainval_len)
        f.create_dataset('opt_train', dtype='uint32', data=options_trainval)
        f.create_dataset('opt_length_train', dtype='uint32', data=options_trainval_len)
        f.create_dataset('opt_list_train', dtype='uint32', data=options_trainval_list)
        f.create_dataset('img_pos_train', dtype='uint32', data=images_trainval_index)

        f.create_dataset('ques_test', dtype='uint32', data=questions_test)
        f.create_dataset('ques_length_test', dtype='uint32', data=questions_test_len)
        f.create_dataset('ans_test', dtype='uint32', data=answers_test)
        f.create_dataset('ans_length_test', dtype='uint32', data=answers_test_len)
        f.create_dataset('cap_test', dtype='uint32', data=captions_test)
        f.create_dataset('cap_length_test', dtype='uint32', data=captions_test_len)
        f.create_dataset('opt_test', dtype='uint32', data=options_test)
        f.create_dataset('opt_length_test', dtype='uint32', data=options_test_len)
        f.create_dataset('opt_list_test', dtype='uint32', data=options_test_list)
        f.create_dataset('img_pos_test', dtype='uint32', data=images_test_index)
        f.create_dataset('num_rounds_test', dtype='uint32', data=num_rounds_test)

    f.close()

    out = {}
    out['ind2word'] = ind2word
    out['word2ind'] = word2ind
    if args.train_split == 'train':
        out['unique_img_train'] = images_train_list
        out['unique_img_val'] = images_val_list
    elif args.train_split == 'trainval':
        out['unique_img_train'] = images_trainval_list
        out['unique_img_test'] = images_test_list

    json.dump(out, open(args.output_json, 'w'))
