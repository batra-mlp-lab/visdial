import os
import json
import h5py
import argparse
import numpy as np
from nltk.tokenize import word_tokenize

def tokenize_data(data, word_count=False):
    '''
    Tokenize captions, questions and answers
    Also maintain word count if required
    '''
    res, word_counts = {}, {}

    print data['split']
    print 'Tokenizing captions...'
    for i in data['data']['dialogs']:
        img_id = i['image_id']
        caption = word_tokenize(i['caption'])
        res[img_id] = {'caption': caption}

    print 'Tokenizing questions...'
    ques_toks, ans_toks = [], []
    for i in data['data']['questions']:
        ques_toks.append(word_tokenize(i + '?'))
    print 'Tokenizing answers...'
    for i in data['data']['answers']:
        ans_toks.append(word_tokenize(i))

    for i in data['data']['dialogs']:
        res[i['image_id']]['dialog'] = i['dialog']
        for j in range(10):
            question = ques_toks[i['dialog'][j]['question']]
            answer = ans_toks[i['dialog'][j]['answer']]
            if word_count == True:
                for word in question + answer:
                    word_counts[word] = word_counts.get(word, 0) + 1

    return res, ques_toks, ans_toks, word_counts

def encode_vocab(data_toks, ques_toks, ans_toks, word2ind):
    '''
    Converts string tokens to indices based on given dictionary
    '''
    max_ques_len, max_ans_len, max_cap_len = 0, 0, 0
    for k, v in data_toks.items():
        image_id = k
        caption = [word2ind.get(word, word2ind['UNK']) \
                for word in v['caption']]
        if max_cap_len < len(caption): max_cap_len = len(caption)
        data_toks[k]['caption_inds'] = caption
        data_toks[k]['caption_len'] = len(caption)

    ques_inds, ans_inds = [], []
    for i in ques_toks:
        question = [word2ind.get(word, word2ind['UNK']) \
                for word in i]
        ques_inds.append(question)

    for i in ans_toks:
        answer = [word2ind.get(word, word2ind['UNK']) \
                for word in i]
        ans_inds.append(answer)

    return data_toks, ques_inds, ans_inds

def create_data_mats(data_toks, ques_inds, ans_inds, params, data_split):
    num_threads = len(data_toks.keys())
    num_rounds = 10
    max_cap_len = params.max_cap_len
    max_ques_len = params.max_ques_len
    max_ans_len = params.max_ans_len

    captions = np.zeros([num_threads, max_cap_len])
    questions = np.zeros([num_threads, num_rounds, max_ques_len])
    answers = np.zeros([num_threads, num_rounds, max_ans_len])

    caption_len = np.zeros(num_threads, dtype=np.int)
    question_len = np.zeros([num_threads, num_rounds], dtype=np.int)
    answer_len = np.zeros([num_threads, num_rounds], dtype=np.int)

    image_index = np.zeros(num_threads)

    answer_index = np.zeros([num_threads, num_rounds])
    options = np.zeros([num_threads, num_rounds, 100])

    image_list = []
    for i in range(num_threads):
        image_id = data_toks.keys()[i]
        image_list.append(image_id)
        image_index[i] = i
        caption_len[i] = len(data_toks[image_id]['caption_inds'][0:max_cap_len])
        captions[i][0:caption_len[i]] = data_toks[image_id]['caption_inds'][0:max_cap_len]
        for j in range(10):
            question_len[i][j] = len(ques_inds[data_toks[image_id]['dialog'][j]['question']][0:max_ques_len])
            questions[i][j][0:question_len[i][j]] = ques_inds[data_toks[image_id]['dialog'][j]['question']][0:max_ques_len]
            answer_len[i][j] = len(ans_inds[data_toks[image_id]['dialog'][j]['answer']][0:max_ans_len])
            answers[i][j][0:answer_len[i][j]] = ans_inds[data_toks[image_id]['dialog'][j]['answer']][0:max_ans_len]

            if data_split in [ 'train', 'val' ]:
                answer_index[i][j] = data_toks[image_id]['dialog'][j]['gt_index'] + 1
                options[i][j] = np.array(data_toks[image_id]['dialog'][j]['answer_options']) + 1

    options_list = np.zeros([len(ans_inds), max_ans_len])
    options_len = np.zeros(len(ans_inds), dtype=np.int)

    if data_split in [ 'train', 'val' ]:
        for i in range(len(ans_inds)):
            options_len[i] = len(ans_inds[i][0:max_ans_len])
            options_list[i][0:options_len[i]] = ans_inds[i][0:max_ans_len]

    return captions, caption_len, questions, question_len, answers, answer_len, options, options_list, options_len, answer_index, image_index, image_list

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-download', default=0, type=int, help='Whether to download VisDial v0.9 data')
    parser.add_argument('-train_split', default='train', help='Choose the data split: train | trainval')


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

    print 'Reading json...'
    data_train = json.load(open(args.input_json_train, 'r'))
    data_val = json.load(open(args.input_json_val, 'r'))
    if args.train_split == 'trainval':
        data_test = json.load(open(args.input_json_test, 'r'))

    # Tokenizing
    data_train_toks, ques_train_toks, ans_train_toks, word_counts_train = tokenize_data(data_train, True)
    if args.train_split == 'train':
        data_val_toks, ques_val_toks, ans_val_toks, _ = tokenize_data(data_val)
    elif args.train_split == 'trainval':
        data_val_toks, ques_val_toks, ans_val_toks, word_counts_val = tokenize_data(data_val, True)
        data_test_toks, ques_test_toks, ans_test_toks, _ = tokenize_data(data_test)

    word_counts_all = dict(word_counts_train)

    if args.train_split == 'trainval':
        # combining the word counts of train and val splits
        for word, count in word_counts_val.items():
            word_counts_all[word] = word_counts_all.get(word, 0) + count

    print 'Building vocabulary...'
    word_counts_all['UNK'] = args.word_count_threshold
    vocab = [word for word in word_counts_all \
            if word_counts_all[word] >= args.word_count_threshold]
    print 'Words: %d' % len(vocab)
    word2ind = {word:word_ind+1 for word_ind, word in enumerate(vocab)}
    ind2word = {word_ind:word for word, word_ind in word2ind.items()}

    print 'Encoding based on vocabulary...'
    data_train_toks, ques_train_inds, ans_train_inds = encode_vocab(data_train_toks, ques_train_toks, ans_train_toks, word2ind)
    data_val_toks, ques_val_inds, ans_val_inds = encode_vocab(data_val_toks, ques_val_toks, ans_val_toks, word2ind)
    if args.train_split == 'trainval':
        data_test_toks, ques_test_inds, ans_test_inds = encode_vocab(data_test_toks, ques_test_toks, ans_test_toks, word2ind)

    print 'Creating data matrices...'
    captions_train, captions_train_len, questions_train, questions_train_len, answers_train, answers_train_len, options_train, options_train_list, options_train_len, answers_train_index, images_train_index, images_train_list = create_data_mats(data_train_toks, ques_train_inds, ans_train_inds, args, 'train')
    captions_val, captions_val_len, questions_val, questions_val_len, answers_val, answers_val_len, options_val, options_val_list, options_val_len, answers_val_index, images_val_index, images_val_list = create_data_mats(data_val_toks, ques_val_inds, ans_val_inds, args, 'val')
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

        captions_test, captions_test_len, questions_test, questions_test_len, answers_test, answers_test_len, _, _, _, _, images_test_index, images_test_list = create_data_mats(data_test_toks, ques_test_inds, ans_test_inds, args, 'test')

    print 'Saving hdf5...'
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
        f.create_dataset('ques_trainval', dtype='uint32', data=questions_trainval)
        f.create_dataset('ques_length_trainval', dtype='uint32', data=questions_trainval_len)
        f.create_dataset('ans_trainval', dtype='uint32', data=answers_trainval)
        f.create_dataset('ans_length_trainval', dtype='uint32', data=answers_trainval_len)
        f.create_dataset('ans_index_trainval', dtype='uint32', data=answers_trainval_index)
        f.create_dataset('cap_trainval', dtype='uint32', data=captions_trainval)
        f.create_dataset('cap_length_trainval', dtype='uint32', data=captions_trainval_len)
        f.create_dataset('opt_trainval', dtype='uint32', data=options_trainval)
        f.create_dataset('opt_length_trainval', dtype='uint32', data=options_trainval_len)
        f.create_dataset('opt_list_trainval', dtype='uint32', data=options_trainval_list)
        f.create_dataset('img_pos_trainval', dtype='uint32', data=images_trainval_index)

        f.create_dataset('ques_test', dtype='uint32', data=questions_test)
        f.create_dataset('ques_length_test', dtype='uint32', data=questions_test_len)
        f.create_dataset('ans_test', dtype='uint32', data=answers_test)
        f.create_dataset('ans_length_test', dtype='uint32', data=answers_test_len)
        f.create_dataset('cap_test', dtype='uint32', data=captions_test)
        f.create_dataset('cap_length_test', dtype='uint32', data=captions_test_len)
        f.create_dataset('img_pos_test', dtype='uint32', data=images_test_index)

    f.close()

    out = {}
    out['ind2word'] = ind2word
    out['word2ind'] = word2ind
    if args.train_split == 'train':
        out['unique_img_train'] = images_train_list
        out['unique_img_val'] = images_val_list
    elif args.train_split == 'trainval':
        out['unique_img_trainval'] = images_trainval_list
        out['unique_img_test'] = images_test_list

    json.dump(out, open(args.output_json, 'w'))
