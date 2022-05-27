from ml.datasets.flights import FlightHoloClean

column_id = 5


#data = BlackOakDataSetUppercase()
#data = HospitalHoloClean()
data = FlightHoloClean()
data.name = 'Flights'

command = 'python scripts/preprocess.py \\\n' + \
          '--input_txt /root/torch-rnn/storage/' + data.name + '/column_' + str(column_id) + '/orig_input/column_' + str(column_id) + '.txt \\\n' + \
  '--output_h5 /root/torch-rnn/storage/' + data.name + '/column_' + str(column_id) + '/input/my_data.h5 \\\n' + \
  '--output_json /root/torch-rnn/storage/' + data.name + '/column_' + str(column_id) + '/input/my_data.json\n\n' + \
  'th train.lua -input_h5 /root/torch-rnn/storage/' + data.name + '/column_' + str(column_id) + '/input/my_data.h5 -input_json /root/torch-rnn/storage/' + data.name + '/column_' + str(column_id) + '/input/my_data.json -checkpoint_name /root/torch-rnn/storage/' + data.name + '/column_' + str(column_id) + '/cv/checkpoint -checkpoint_every 100 -rnn_size 128 -num_layers 1 -batch_size 10\n\n' + \
  'th show_activation.lua -length 100 -gpu 0 -gpu_backend cuda -file /root/torch-rnn/storage/' + data.name + '/column_' + str(column_id) + '/orig_input/column_' + str(column_id) + '.txt -output /root/torch-rnn/storage/' + data.name + '/column_' + str(column_id) + '/features/  -checkpoint /root/torch-rnn/storage/' + data.name + '/column_' + str(column_id) + '/cv/checkpoint_\n\n' + \
  'python DeepFeatures.py -d /root/torch-rnn/storage/' + data.name + ' -a 2 -c ' + str(column_id) + ' -o /root/torch-rnn/storage/' + data.name + '_avg_state/out' + str(column_id) + '.npz\n\n' + \
  'python DeepFeatures.py -d /root/torch-rnn/storage/' + data.name + ' -a 1 -c ' + str(column_id) + ' -o /root/torch-rnn/storage/' + data.name + '_last_state/out' + str(column_id) + '.npz\n\n' + \
  'rm -r /root/torch-rnn/storage/' + data.name + '/column_' + str(column_id) + '/features/*'


print command