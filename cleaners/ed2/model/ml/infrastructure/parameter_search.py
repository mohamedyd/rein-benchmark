import os
import glob
import shutil
import json
import numpy as np
import sys

def evaluate_check_point_json(checkpoint_json_file):

    with open(checkpoint_json_file) as data_file:
        data = json.load(data_file)

    loss_history = data['val_loss_history']
    checkpoint_pointer = data['val_loss_history_it']

    best_i = np.argmin(loss_history)

    return loss_history[best_i], checkpoint_pointer[best_i]

def get_latest_checkpoint(path):
    try:
        newest = max(glob.iglob(path + '/*.json'), key=os.path.getctime)
        return newest
    except ValueError:
        return None

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


def get_best_loss(best_loss, json_file):
    if os.path.exists(json_file):
        best_loss_new, _ = evaluate_check_point_json(json_file)
        return best_loss_new
    else:
        return best_loss


run = True

data_name = 'BlackOakUppercase'
num_columns = 12

for column_id in range(num_columns):
    command = 'python scripts/preprocess.py \\\n' + \
              '--input_txt /root/torch-rnn/storage/' + data_name + '/column_' + str(
        column_id) + '/orig_input/column_' + str(column_id) + '.txt \\\n' + \
              '--output_h5 /root/torch-rnn/storage/' + data_name + '/column_' + str(
        column_id) + '/input/my_data.h5 \\\n' + \
              '--output_json /root/torch-rnn/storage/' + data_name + '/column_' + str(
        column_id) + '/input/my_data.json\n\n'

    run_command(command, run)


    directory = '/root/torch-rnn/storage/' + data_name + '/column_' + str(column_id) + '/best'

    if not os.path.exists(directory):
        os.makedirs(directory)

    best_loss = sys.float_info.max

    # check whether we need batch size of 50
    # check whether seq_length was important

    for units in [128]:
        for num_layers in [1]:
            for batch_size in [5, 10]:
                for learning_rate in [0.001, 0.002, 0.003]:
                    for dropout in [0.0, 0.1, 0.3]:
                        for seq_length in [15, 25, 50]:

                            command = 'th train.lua ' + \
                                  '-input_h5 /root/torch-rnn/storage/' + data_name + '/column_' + str(column_id) + '/input/my_data.h5 ' + \
                                  '-input_json /root/torch-rnn/storage/' + data_name + '/column_' + str(column_id) + '/input/my_data.json '+ \
                                  '-checkpoint_name /root/torch-rnn/storage/' + data_name + '/column_' + str(column_id) + '/cv/checkpoint '+ \
                                  '-rnn_size ' + str(units) + ' ' + \
                                  '-checkpoint_every 50 ' + \
                                  '-num_layers ' + str(num_layers) + ' ' + \
                                  '-dropout ' + str(dropout) + ' ' + \
                                  '-seq_length ' + str(seq_length) + ' ' + \
                                  '-max_epochs 100 ' + \
                                  '-batch_size ' + str(batch_size) + ' ' + \
                                  '-learning_rate ' + str(learning_rate) + \
                                  '\n\n'

                            run_command(command, run)

                            checkpoint_path = '/root/torch-rnn/storage/' + data_name + '/column_' + str(column_id) + '/cv'
                            latest_checkpoint_file = get_latest_checkpoint(checkpoint_path)

                            if latest_checkpoint_file == None:
                                with open(directory + "/log.txt", "a") as myfile:
                                    myfile.write("rnn_size: " + str(units) + ", " + \
                                                 "num_layers: " + str(num_layers) + ", " + \
                                                 "dropout: " + str(dropout) + ", " + \
                                                 "seq_length: " + str(seq_length) + ", " + \
                                                 "batch_size: " + str(batch_size) + ", " + \
                                                 "learning_rate: " + str(learning_rate) + ", " + \
                                                 "best checkpoint id: " + "none" + ", " + \
                                                 "loss: " + "none" + "\n"
                                                 )
                            else:
                                loss, checkpoint_index = evaluate_check_point_json(latest_checkpoint_file)

                                best_loss = get_best_loss(best_loss, directory + "/best.json")

                                if best_loss > loss:
                                    # found a better parameter config
                                    best_loss = loss

                                    # save this checkpoint
                                    shutil.copy(checkpoint_path + "/checkpoint_" + str(checkpoint_index) + ".t7", directory + "/best.t7")
                                    shutil.copy(checkpoint_path + "/checkpoint_" + str(checkpoint_index) + ".json", directory + "/best.json")

                                # log everything
                                with open(directory + "/log.txt", "a") as myfile:
                                    myfile.write("rnn_size: " + str(units) + ", " + \
                                                 "num_layers: "+ str(num_layers) + ", " + \
                                                 "dropout: " + str(dropout) + ", " + \
                                                 "seq_length: " + str(seq_length) + ", " + \
                                                 "batch_size: " + str(batch_size) + ", " + \
                                                 "learning_rate: " + str(learning_rate) + ", " + \
                                                 "best checkpoint id: " + str(checkpoint_index) + ", " + \
                                                 "loss: " + str(loss) + "\n"
                                                 )

                            #clean up old checkpoints
                            delete_folder_content(checkpoint_path)

