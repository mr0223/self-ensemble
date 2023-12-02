DEVICES=$1      # Specify GPU device to use

CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
    --output_path output/baseline/b10.train.csv --num_beams 10 --num_returns 10 --data_type train --batch_size 8 ;
CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
    --output_path output/baseline/b10.valid.csv --num_beams 10 --num_returns 10 --data_type valid --batch_size 8 ;

python shape_predict_input.py --num_beams 10 --data_type train ;
python shape_predict_input.py --num_beams 10 --data_type valid ;

CUDA_VISIBLE_DEVICES=${DEVICES} python train_predict_model.py --num_beams 10 ;
