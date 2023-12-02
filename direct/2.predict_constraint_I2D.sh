DEVICES=$1                      # Specify GPU device to use
PREDICT_MODEL_CHECKPOINT=$2     # Specify the name of the predict model trained in 1.train_predict_model.sh

CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
    --checkpoint ${BART_CHECKPOINT} --input_path input/test.indirect --output_path output/baseline/b40.test.direct.csv --num_beams 40 --num_returns 40 ;
CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
    --checkpoint ${BART_CHECKPOINT} --input_path input/test.indirect --output_path output/baseline/b60.test.direct.csv --num_beams 60 --num_returns 60 ;
CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
    --checkpoint ${BART_CHECKPOINT} --input_path input/test.indirect --output_path output/baseline/b100.test.direct.csv --num_beams 100 --num_returns 100 ;

python shape_predict_input.py \
    --input_path input/test.indirect --baseline_output_path output/baseline/b40.test.direct.csv \
    --ref_path input/test.direct --predict_input_path predict_input/b40.test.direct.csv --num_beams 40 ;
python shape_predict_input.py \
    --input_path input/test.indirect --baseline_output_path output/baseline/b60.test.direct.csv \
    --ref_path input/test.direct --predict_input_path predict_input/b60.test.direct.csv --num_beams 60 ;
python shape_predict_input.py \
    --input_path input/test.indirect --baseline_output_path output/baseline/b100.test.direct.csv \
    --ref_path input/test.direct --predict_input_path predict_input/b100.test.direct.csv --num_beams 100 ;

CUDA_VISIBLE_DEVICES=${DEVICES} python predict_constraint.py \
    --return_num 40 --data_type test --output_ext direct --checkpoint ${PREDICT_MODEL_CHECKPOINT} ;
CUDA_VISIBLE_DEVICES=${DEVICES} python predict_constraint.py \
    --return_num 60 --data_type test --output_ext direct --checkpoint ${PREDICT_MODEL_CHECKPOINT} ;

python predict_oracle_constraint.py \
    --return_num 100 --data_type test --output_ext direct --checkpoint ${PREDICT_MODEL_CHECKPOINT} ;
