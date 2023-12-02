DEVICES=$1              # Specify GPU device to use
BART_CHECKPOINT=$2      # Specify checkpoints for BART models trained on the direct corpus

mkdir -p output/baseline ;
mkdir -p predict_input ;
mkdir -p predict_model ;

CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
    --checkpoint ${BART_CHECKPOINT} --input_path input/train.w.indirect --output_path output/baseline/b50.train.w.direct.csv --num_beams 50 --num_returns 50 ;
CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
    --checkpoint ${BART_CHECKPOINT} --input_path input/valid.w.indirect --output_path output/baseline/b50.valid.w.direct.csv --num_beams 50 --num_returns 50 ;

python shape_predict_input.py \
    --input_path input/train.w.indirect --baseline_output_path output/baseline/b50.train.w.direct.csv \
    --ref_path input/train.direct --predict_input_path predict_input/b50.train.w.direct.csv --num_beams 50 ;
python shape_predict_input.py \
    --input_path input/valid.w.indirect --baseline_output_path output/baseline/b50.valid.w.direct.csv \
    --ref_path input/valid.direct --predict_input_path predict_input/b50.valid.w.direct.csv --num_beams 50 ;

CUDA_VISIBLE_DEVICES=${DEVICES} python train_predict_model_w.py --num_beams 50 --task I2D ;
