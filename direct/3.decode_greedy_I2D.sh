DEVICES=$1              # Specify GPU device to use
BART_CHECKPOINT=$2      # Specify checkpoints for BART models trained on the direct corpus

CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
  --input_path input/test.indirect \
  --output_file output/both/b40.majority.s1.a0.5.n10.p4.b4.test.direct \
  --constraint constraint/both/b40.majority.test.direct.pkl \
  --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
  --length_penalty 0.6 \
  --prune_factor 4 --sat_tolerance 10 --beta 0.25 \
  --look_ahead_step 1 --alpha 0.5 --look_ahead_width 1 ;

CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
  --input_path input/test.indirect \
  --output_file output/positive/b40.majority.s1.a0.01.n3.p4.b4.test.direct \
  --constraint constraint/positive/b40.majority.test.direct.pkl \
  --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
  --length_penalty 0.6 \
  --prune_factor 4 --sat_tolerance 3 --beta 0.25 \
  --look_ahead_step 1 --alpha 0.01 --look_ahead_width 1 ;

CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
  --input_path input/test.indirect \
  --output_file output/negative/b60.majority.s1.a0.01.n3.p1.b4.test.direct \
  --constraint constraint/negative/b60.majority.test.direct.pkl \
  --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
  --length_penalty 0.6 \
  --prune_factor 1 --sat_tolerance 3 --beta 0.25 \
  --look_ahead_step 1 --alpha 0.01 --look_ahead_width 1 ;


CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
  --input_path input/test.indirect \
  --output_file output/_both/b100.majority.s1.a0.02.n1.p100.b4.test.direct \
  --constraint constraint/_both/b100.majority.test.direct.pkl \
  --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
  --length_penalty 0.6 \
  --prune_factor 100 --sat_tolerance 1 --beta 0.25 \
  --look_ahead_step 1 --alpha 0.02 --look_ahead_width 1 ;

CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
  --input_path input/test.indirect \
  --output_file output/_positive/b100.majority.s1.a0.05.n1.p20.b4.test.direct \
  --constraint constraint/_positive/b100.majority.test.direct.pkl \
  --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
  --length_penalty 0.6 \
  --prune_factor 20 --sat_tolerance 1 --beta 0.25 \
  --look_ahead_step 1 --alpha 0.05 --look_ahead_width 1 ;

CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
  --input_path input/test.indirect \
  --output_file output/_negative/b100.majority.s1.a0.05.n1.p40.b4.test.direct \
  --constraint constraint/_negative/b100.majority.test.direct.pkl \
  --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
  --length_penalty 0.6 \
  --prune_factor 40 --sat_tolerance 1 --beta 0.25 \
  --look_ahead_step 1 --alpha 0.05 --look_ahead_width 1 ;
