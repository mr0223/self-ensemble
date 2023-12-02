DEVICES=$1    # Specify GPU device to use
MODEL_NAME='facebook/bart-large-cnn'

CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${MODEL_NAME} \
  --data_type test \
  --output_file output/both/b90.majority.s1.a0.05.n8.p2.b6.test \
  --constraint constraint/both/b90.majority.test.pkl \
  --batch_size 1 --beam_size 6 --max_tgt_length 1024 --min_tgt_length 3 \
  --length_penalty 0.6 \
  --prune_factor 2 --sat_tolerance 8 --beta 0.25 \
  --look_ahead_step 1 --alpha 0.05 --look_ahead_width 1 ;

CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${MODEL_NAME} \
  --data_type test \
  --output_file output/positive/b10.majority.s1.a0.01.n3.p1.b6.test \
  --constraint constraint/positive/b10.majority.test.pkl \
  --batch_size 1 --beam_size 6 --max_tgt_length 1024 --min_tgt_length 3 \
  --length_penalty 0.6 \
  --prune_factor 1 --sat_tolerance 3 --beta 0.25 \
  --look_ahead_step 1 --alpha 0.01 --look_ahead_width 1 ;

CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${MODEL_NAME} \
  --data_type test \
  --output_file output/negative/b90.majority.s1.a0.01.n15.p2.b6.test \
  --constraint constraint/negative/b90.majority.test.pkl \
  --batch_size 1 --beam_size 6 --max_tgt_length 1024 --min_tgt_length 3 \
  --length_penalty 0.6 \
  --prune_factor 2 --sat_tolerance 15 --beta 0.25 \
  --look_ahead_step 1 --alpha 0.01 --look_ahead_width 1 ;


CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${MODEL_NAME} \
  --data_type test \
  --output_file output/_both/b100.majority.s1.a0.02.n1.p50.b6.test \
  --constraint constraint/_both/b100.majority.test.pkl \
  --batch_size 1 --beam_size 6 --max_tgt_length 1024 --min_tgt_length 3 \
  --length_penalty 0.6 \
  --prune_factor 50 --sat_tolerance 1 --beta 0.25 \
  --look_ahead_step 1 --alpha 0.02 --look_ahead_width 1 ;

CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${MODEL_NAME} \
  --data_type test \
  --output_file output/_positive/b100.majority.s1.a0.05.n1.p20.b6.test \
  --constraint constraint/_positive/b100.majority.test.pkl \
  --batch_size 1 --beam_size 6 --max_tgt_length 1024 --min_tgt_length 3 \
  --length_penalty 0.6 \
  --prune_factor 20 --sat_tolerance 1 --beta 0.25 \
  --look_ahead_step 1 --alpha 0.05 --look_ahead_width 1 ;

CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${MODEL_NAME} \
  --data_type test \
  --output_file output/_negative/b100.majority.s1.a0.05.n1.p100.b6.test \
  --constraint constraint/_negative/b100.majority.test.pkl \
  --batch_size 1 --beam_size 6 --max_tgt_length 1024 --min_tgt_length 3 \
  --length_penalty 0.6 \
  --prune_factor 100 --sat_tolerance 1 --beta 0.25 \
  --look_ahead_step 1 --alpha 0.05 --look_ahead_width 1 ;
