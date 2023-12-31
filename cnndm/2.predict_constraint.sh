DEVICES=$1                      # Specify GPU device to use
PREDICT_MODEL_CHECKPOINT=$2     # Specify the name of the predict model trained in 1.train_predict_model.sh

mkdir -p constraint/both ;
mkdir -p constraint/positive ;
mkdir -p constraint/negative ;
mkdir -p constraint/_both ;
mkdir -p constraint/_positive ;
mkdir -p constraint/_negative ;

CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
    --output_path output/baseline/b10.test.csv --num_beams 10 --num_returns 10 --data_type test --batch_size 8 ;
CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
    --output_path output/baseline/b80.test.csv --num_beams 80 --num_returns 80 --data_type test --batch_size 8 ;
CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
    --output_path output/baseline/b100.test.csv --num_beams 100 --num_returns 100 --data_type test --batch_size 8 ;

python shape_predict_input.py --num_beams 10 --data_type test ;
python shape_predict_input.py --num_beams 80 --data_type test ;
python shape_predict_input.py --num_beams 100 --data_type test ;

CUDA_VISIBLE_DEVICES=${DEVICES} python predict_constraint.py \
    --return_num 10 --data_type test --checkpoint ${PREDICT_MODEL_CHECKPOINT} ;
CUDA_VISIBLE_DEVICES=${DEVICES} python predict_constraint.py \
    --return_num 80 --data_type test --checkpoint ${PREDICT_MODEL_CHECKPOINT} ;
CUDA_VISIBLE_DEVICES=${DEVICES} python predict_constraint.py \
    --return_num 100 --data_type test --checkpoint ${PREDICT_MODEL_CHECKPOINT} ;

python predict_oracle_constraint.py \
    --return_num 100 --data_type test --checkpoint ${PREDICT_MODEL_CHECKPOINT} ;
