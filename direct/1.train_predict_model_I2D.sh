DEVICES=$1              # Specify GPU device to use
BART_CHECKPOINT=$2      # Specify checkpoints for BART models trained on the direct corpus

CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
    --checkpoint ${BART_CHECKPOINT} --input_path input/train.indirect --output_path output/baseline/b50.train.direct.csv --num_beams 50 --num_returns 50 ;
CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
    --checkpoint ${BART_CHECKPOINT} --input_path input/valid.indirect --output_path output/baseline/b50.valid.direct.csv --num_beams 50 --num_returns 50 ;

python shape_predict_input.py \
    --input_path input/train.indirect --baseline_output_path output/baseline/b50.train.direct.csv \
    --ref_path input/train.direct --predict_input_path predict_input/b50.train.direct.csv --num_beams 50 ;
python shape_predict_input.py \
    --input_path input/valid.indirect --baseline_output_path output/baseline/b50.valid.direct.csv \
    --ref_path input/valid.direct --predict_input_path predict_input/b50.valid.direct.csv --num_beams 50 ;

CUDA_VISIBLE_DEVICES=${DEVICES} python train_predict_model.py --num_beams 50 --task I2D ;
