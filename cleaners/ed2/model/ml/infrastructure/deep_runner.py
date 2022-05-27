import os
import glob
import shutil
import json
from pprint import pprint
import numpy as np

def evaluate_check_point_json(checkpoint_json_file):

    with open(checkpoint_json_file) as data_file:
        data = json.load(data_file)

    loss_history = data['val_loss_history']
    checkpoint_pointer = data['val_loss_history_it']

    best_i = np.argmin(loss_history)

    return loss_history[best_i], checkpoint_pointer[best_i]

def get_latest_checkpoint(path):
    newest = max(glob.iglob(path + '/*.json'), key=os.path.getctime)
    return newest

def delete_folder_content(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path): shutil.rmtree(file_path)
        except Exception as e:
            print(e)


def run_command(command, run):
    print command
    if run:
        os.system(command)


run = False

num_layers = 2
batch_size = 50


data_name = 'BlackOakUppercase'
num_columns = 12

for column_id in range(num_columns):

    command = 'python scripts/preprocess.py \\\n' + \
              '--input_txt /root/torch-rnn/storage/' + data_name + '/column_' + str(column_id) + '/orig_input/column_' + str(column_id) + '.txt \\\n' + \
      '--output_h5 /root/torch-rnn/storage/' + data_name + '/column_' + str(column_id) + '/input/my_data.h5 \\\n' + \
      '--output_json /root/torch-rnn/storage/' + data_name + '/column_' + str(column_id) + '/input/my_data.json\n\n'

    run_command(command, run)

    command = 'th train.lua -input_h5 /root/torch-rnn/storage/' + data_name + '/column_' + str(
        column_id) + '/input/my_data.h5 -input_json /root/torch-rnn/storage/' + data_name + '/column_' + str(
        column_id) + '/input/my_data.json -checkpoint_name /root/torch-rnn/storage/' + data_name + '/column_' + str(
        column_id) + '/cv/checkpoint -rnn_size 128 ' + '-checkpoint_every 100 ' + \
        '-num_layers ' + str(num_layers) + ' -batch_size ' + str(batch_size) + '\n\n'

    run_command(command, run)

    checkpoint_path = '/root/torch-rnn/storage/' + data_name + '/column_' + str(column_id) + '/cv'
    latest_checkpoint_file = get_latest_checkpoint(checkpoint_path)

    _, checkpoint_index = evaluate_check_point_json(latest_checkpoint_file)

    best_checkpoint = checkpoint_path + '/checkpoint_' + str(checkpoint_index) + '.t7'

    command = 'th show_activation.lua -length 100 -gpu 0 -gpu_backend cuda -file /root/torch-rnn/storage/' + data_name + '/column_' + str(column_id) + '/orig_input/column_' + str(column_id) + '.txt -output /root/torch-rnn/storage/' + data_name + '/column_' + str(column_id) + '/features/  -checkpoint ' + str(best_checkpoint) + '\n\n'
    run_command(command, run)

    command = 'python DeepFeatures.py -d /root/torch-rnn/storage/' + data_name + ' -a 2 -c ' + str(column_id) + ' -o /root/torch-rnn/storage/' + data_name + '_avg_state/out' + str(column_id) + '.npz\n\n'
    run_command(command, run)

    command = 'python DeepFeatures.py -d /root/torch-rnn/storage/' + data_name + ' -a 1 -c ' + str(column_id) + ' -o /root/torch-rnn/storage/' + data_name + '_last_state/out' + str(column_id) + '.npz\n\n'
    run_command(command, run)

    if run:
        delete_folder_content('/root/torch-rnn/storage/' + data_name + '/column_' + str(column_id) + '/features')
