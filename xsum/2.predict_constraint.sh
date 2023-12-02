DEVICES=$1                      # Specify GPU device to use
PREDICT_MODEL_CHECKPOINT=$2     # Specify the name of the predict model trained in 1.train_predict_model.sh

CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
    --output_path output/baseline/b10.test.csv --num_beams 10 --num_returns 10 --data_type test --batch_size 8 ;
CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
    --output_path output/baseline/b90.test.csv --num_beams 90 --num_returns 90 --data_type test --batch_size 8 ;
CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
    --output_path output/baseline/b100.test.csv --num_beams 100 --num_returns 100 --data_type test --batch_size 8 ;

python shape_predict_input.py --num_beams 10 --data_type test ;
python shape_predict_input.py --num_beams 90 --data_type test ;
python shape_predict_input.py --num_beams 100 --data_type test ;

CUDA_VISIBLE_DEVICES=${DEVICES} python predict_constraint.py \
    --return_num 10 --data_type test --checkpoint ${PREDICT_MODEL_CHECKPOINT} ;
CUDA_VISIBLE_DEVICES=${DEVICES} python predict_constraint.py \
    --return_num 90 --data_type test --checkpoint ${PREDICT_MODEL_CHECKPOINT} ;

python predict_oracle_constraint.py \
    --return_num 100 --data_type test --checkpoint ${PREDICT_MODEL_CHECKPOINT} ;
