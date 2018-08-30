import tensorflow as tf
import vis_lstm_model
import data_loader
import argparse
import numpy as np
from os.path import isfile, join
from PIL import Image
import utils
import re

DATA_PATH = 'Data'
MODEL_PATH = 'Data/Models/model7.ckpt'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type = str, default = MODEL_PATH,
                       help = 'Model Path')
    parser.add_argument('--num_lstm_layers', type = int, default = 2,
                       help = 'num_lstm_layers')
    parser.add_argument('--feature_length', type = int, default = 4096,
                       help = 'feature_length')
    parser.add_argument('--rnn_size', type = int, default = 512,
                       help = 'rnn_size')
    parser.add_argument('--embedding_size', type = int, default = 512,
                       help = 'embedding_size'),
    parser.add_argument('--word_emb_dropout', type = float, default = 1.0,
                       help = 'word_emb_dropout')
    parser.add_argument('--image_dropout', type = float, default = 1.0,
                       help = 'image_dropout')
    parser.add_argument('--data_dir', type = str, default = DATA_PATH,
                       help = 'Data directory')
    parser.add_argument('--image_features', type = str, default = 'vgg16',
                       help = 'Image features')
    args = parser.parse_args()

    vocab_data = data_loader.get_question_answer_vocab(args.data_dir)

    model_options = {
		'num_lstm_layers' : args.num_lstm_layers,
		'rnn_size' : args.rnn_size,
		'embedding_size' : args.embedding_size,
		'word_emb_dropout' : args.word_emb_dropout,
		'image_dropout' : args.image_dropout,
		'feature_length' : args.feature_length,
		'lstm_steps' : vocab_data['max_question_length'] + 1,
		'q_vocab_size' : len(vocab_data['question_vocab']),
		'ans_vocab_size' : len(vocab_data['answer_vocab'])
	}
    ans_map = { vocab_data['answer_vocab'][ans] : ans for ans in vocab_data['answer_vocab']}
    question_vocab = vocab_data['question_vocab']
    word_regex = re.compile(r'\w+')

    print("Reading QA DATA")
    test_data = data_loader.load_test_questions()
    print(len(test_data['question_vocab']))
    features, image_id_list = data_loader.load_features(args.data_dir, 'test', args.image_features)

    print("Features", features.shape)
    print("Image_id_list", image_id_list.shape)

    image_id_map = {}
    for i in range(len(image_id_list)):
        image_id_map[ image_id_list[i] ] = i

    model = vis_lstm_model.Vis_lstm_model(model_options)
    input_tensors, t_prediction, t_ans_probab = model.build_generator()
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    saver.restore(sess, args.model_path)

    stop_vocab = ['a', 'an', 'the']

    for i, now_image in enumerate(test_data['testing']):
        now_image_path = 'Data/test2015/COCO_test2015_%.12d.jpg'%(now_image['image_id'])
        img = Image.open(now_image_path)
        img.show()
        question_ids = np.zeros((1, vocab_data['max_question_length']), dtype = 'int32')

        print('Question:', now_image['question'])
        question_words = re.findall(word_regex, now_image['question'])
        question_words = list(filter(lambda x: data_loader.vocab_handle(x) not in stop_vocab, question_words))
        base = vocab_data['max_question_length'] - len(question_words)
        for j in range(0, len(question_words)):
            now_question_words = data_loader.vocab_handle(question_words[j])
            if now_question_words in question_vocab:
                question_ids[0][base + j] = question_vocab[now_question_words]
            else:
                question_ids[0][base + j] = question_vocab['UNK']
        
        now_index = image_id_map[ test_data['testing'][i]['image_id'] ]

        pred, answer_probab = sess.run([t_prediction, t_ans_probab], feed_dict={
            input_tensors['features']:features[now_index].reshape(1, args.feature_length),
            input_tensors['sentence']:question_ids,
        })

        print("Ans:", ans_map[pred[0]])
        answer_probab_tuples = [(-answer_probab[0][idx], idx) for idx in range(len(answer_probab[0]))]
        answer_probab_tuples.sort()
        print("Top Answers:")
        for i in range(5):
            print(ans_map[ answer_probab_tuples[i][1] ])
        input()
    sess.close()

if __name__ == '__main__':
	main()