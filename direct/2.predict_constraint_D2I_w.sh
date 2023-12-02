DEVICES=$1                      # Specify GPU device to use
PREDICT_MODEL_CHECKPOINT=$2     # Specify the name of the predict model trained in 1.train_predict_model.sh

mkdir -p constraint/both ;
mkdir -p constraint/positive ;
mkdir -p constraint/negative ;
mkdir -p constraint/_both ;
mkdir -p constraint/_positive ;
mkdir -p constraint/_negative ;

CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
    --checkpoint ${BART_CHECKPOINT} --input_path input/test.w.direct --output_path output/baseline/b20.test.w.indirect.csv --num_beams 20 --num_returns 20 ;
CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
    --checkpoint ${BART_CHECKPOINT} --input_path input/test.w.direct --output_path output/baseline/b100.test.w.indirect.csv --num_beams 100 --num_returns 100 ;

python shape_predict_input.py \
    --input_path input/test.w.direct --baseline_output_path output/baseline/b20.test.w.indirect.csv \
    --ref_path input/test.indirect --predict_input_path predict_input/b20.test.w.indirect.csv --num_beams 20 ;
python shape_predict_input.py \
    --input_path input/test.w.direct --baseline_output_path output/baseline/b100.test.w.indirect.csv \
    --ref_path input/test.indirect --predict_input_path predict_input/b100.test.w.indirect.csv --num_beams 100 ;

CUDA_VISIBLE_DEVICES=${DEVICES} python predict_constraint.py \
    --return_num 20 --data_type test --output_ext w.indirect --checkpoint ${PREDICT_MODEL_CHECKPOINT} ;

python predict_oracle_constraint.py \
    --return_num 100 --data_type test --output_ext w.indirect --checkpoint ${PREDICT_MODEL_CHECKPOINT} ;
