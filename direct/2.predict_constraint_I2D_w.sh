DEVICES=$1                      # Specify GPU device to use
PREDICT_MODEL_CHECKPOINT=$2     # Specify the name of the predict model trained in 1.train_predict_model.sh

mkdir -p constraint/both ;
mkdir -p constraint/positive ;
mkdir -p constraint/negative ;
mkdir -p constraint/_both ;
mkdir -p constraint/_positive ;
mkdir -p constraint/_negative ;

CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
    --checkpoint ${BART_CHECKPOINT} --input_path input/test.w.indirect --output_path output/baseline/b50.test.w.direct.csv --num_beams 50 --num_returns 50 ;
CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
    --checkpoint ${BART_CHECKPOINT} --input_path input/test.w.indirect --output_path output/baseline/b100.test.w.direct.csv --num_beams 100 --num_returns 100 ;

python shape_predict_input.py \
    --input_path input/test.w.indirect --baseline_output_path output/baseline/b50.test.w.direct.csv \
    --ref_path input/test.direct --predict_input_path predict_input/b50.test.w.direct.csv --num_beams 50 ;
python shape_predict_input.py \
    --input_path input/test.w.indirect --baseline_output_path output/baseline/b100.test.w.direct.csv \
    --ref_path input/test.direct --predict_input_path predict_input/b100.test.w.direct.csv --num_beams 100 ;

CUDA_VISIBLE_DEVICES=${DEVICES} python predict_constraint.py \
    --return_num 50 --data_type test --output_ext w.direct --checkpoint ${PREDICT_MODEL_CHECKPOINT} ;

python predict_oracle_constraint.py \
    --return_num 100 --data_type test --output_ext w.direct --checkpoint ${PREDICT_MODEL_CHECKPOINT} ;
