DEVICES=$1    # Specify GPU device to use
MODEL_NAME='facebook/bart-large-cnn'

mkdir -p output/both ;
mkdir -p output/positive ;
mkdir -p output/negative ;
mkdir -p output/_both ;
mkdir -p output/_positive ;
mkdir -p output/_negative ;

CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${MODEL_NAME} \
  --data_type test \
  --output_file output/both/b80.majority.s1.a0.5.n3.p3.b4.test \
  --constraint constraint/both/b80.majority.test.pkl \
  --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
  --length_penalty 0.6 \
  --prune_factor 3 --sat_tolerance 3 --beta 0.25 \
  --look_ahead_step 1 --alpha 0.5 --look_ahead_width 1 ;

CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${MODEL_NAME} \
  --data_type test \
  --output_file output/positive/b10.majority.s1.a0.01.n4.p20.b4.test \
  --constraint constraint/positive/b10.majority.test.pkl \
  --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
  --length_penalty 0.6 \
  --prune_factor 20 --sat_tolerance 4 --beta 0.25 \
  --look_ahead_step 1 --alpha 0.01 --look_ahead_width 1 ;

CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${MODEL_NAME} \
  --data_type test \
  --output_file output/negative/b100.majority.s1.a0.01.n3.p1.b4.test \
  --constraint constraint/negative/b100.majority.test.pkl \
  --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
  --length_penalty 0.6 \
  --prune_factor 1 --sat_tolerance 3 --beta 0.25 \
  --look_ahead_step 1 --alpha 0.01 --look_ahead_width 1 ;


CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${MODEL_NAME} \
  --data_type test \
  --output_file output/_both/b100.majority.s1.a0.02.n1.p100.b4.test \
  --constraint constraint/_both/b100.majority.test.pkl \
  --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
  --length_penalty 0.6 \
  --prune_factor 100 --sat_tolerance 1 --beta 0.25 \
  --look_ahead_step 1 --alpha 0.02 --look_ahead_width 1 ;

CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${MODEL_NAME} \
  --data_type test \
  --output_file output/_positive/b100.majority.s1.a0.05.n1.p20.b4.test \
  --constraint constraint/_positive/b100.majority.test.pkl \
  --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
  --length_penalty 0.6 \
  --prune_factor 20 --sat_tolerance 1 --beta 0.25 \
  --look_ahead_step 1 --alpha 0.05 --look_ahead_width 1 ;

CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${MODEL_NAME} \
  --data_type test \
  --output_file output/_negative/b100.majority.s1.a0.05.n1.p50.b4.test \
  --constraint constraint/_negative/b100.majority.test.pkl \
  --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
  --length_penalty 0.6 \
  --prune_factor 50 --sat_tolerance 1 --beta 0.25 \
  --look_ahead_step 1 --alpha 0.05 --look_ahead_width 1 ;
