DEVICES=$1                      # Specify GPU device to use
PREDICT_MODEL_CHECKPOINT=$2     # Specify the name of the predict model trained in 1.train_predict_model.sh

mkdir -p constraint/both ;
mkdir -p constraint/positive ;
mkdir -p constraint/negative ;
mkdir -p constraint/_both ;
mkdir -p constraint/_positive ;
mkdir -p constraint/_negative ;

CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
    --checkpoint ${BART_CHECKPOINT} --input_path input/test.direct --output_path output/baseline/b10.test.indirect.csv --num_beams 10 --num_returns 10 ;
CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
    --checkpoint ${BART_CHECKPOINT} --input_path input/test.direct --output_path output/baseline/b30.test.indirect.csv --num_beams 30 --num_returns 30 ;
CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
    --checkpoint ${BART_CHECKPOINT} --input_path input/test.direct --output_path output/baseline/b70.test.indirect.csv --num_beams 70 --num_returns 70 ;
CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
    --checkpoint ${BART_CHECKPOINT} --input_path input/test.direct --output_path output/baseline/b100.test.indirect.csv --num_beams 100 --num_returns 100 ;

python shape_predict_input.py \
    --input_path input/test.direct --baseline_output_path output/baseline/b10.test.indirect.csv \
    --ref_path input/test.indirect --predict_input_path predict_input/b10.test.indirect.csv --num_beams 10 ;
python shape_predict_input.py \
    --input_path input/test.direct --baseline_output_path output/baseline/b30.test.indirect.csv \
    --ref_path input/test.indirect --predict_input_path predict_input/b30.test.indirect.csv --num_beams 30 ;
python shape_predict_input.py \
    --input_path input/test.direct --baseline_output_path output/baseline/b70.test.indirect.csv \
    --ref_path input/test.indirect --predict_input_path predict_input/b70.test.indirect.csv --num_beams 70 ;
python shape_predict_input.py \
    --input_path input/test.direct --baseline_output_path output/baseline/b100.test.indirect.csv \
    --ref_path input/test.indirect --predict_input_path predict_input/b100.test.indirect.csv --num_beams 100 ;

CUDA_VISIBLE_DEVICES=${DEVICES} python predict_constraint.py \
    --return_num 10 --data_type test --output_ext indirect --checkpoint ${PREDICT_MODEL_CHECKPOINT} ;
CUDA_VISIBLE_DEVICES=${DEVICES} python predict_constraint.py \
    --return_num 30 --data_type test --output_ext indirect --checkpoint ${PREDICT_MODEL_CHECKPOINT} ;
CUDA_VISIBLE_DEVICES=${DEVICES} python predict_constraint.py \
    --return_num 70 --data_type test --output_ext indirect --checkpoint ${PREDICT_MODEL_CHECKPOINT} ;

python predict_oracle_constraint.py \
    --return_num 100 --data_type test --output_ext indirect --checkpoint ${PREDICT_MODEL_CHECKPOINT} ;
