DEVICES=$1              # Specify GPU device to use
BART_CHECKPOINT=$2      # Specify checkpoints for BART models trained on the direct corpus
TASK=$3                 # Specify a task from "I2D", "D2I", "I2Dw", "D2Iw"
                        # - I2D: Indirect-to-Direct w/o history task
                        # - D2I: Direct-to-Indirect w/o history task
                        # - I2Dw: Indirect-to-Direct w/ history task
                        # - D2Iw: Direct-to-Indirect w/ history task

mkdir -p output/baseline ;
mkdir -p predict_input ;
mkdir -p predict_model ;

if [ ${TASK} = "I2D" ]; then
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
fi

if [ ${TASK} = "D2I" ]; then
    CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
        --checkpoint ${BART_CHECKPOINT} --input_path input/train.direct --output_path output/baseline/b50.train.indirect.csv --num_beams 50 --num_returns 50 ;
    CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
        --checkpoint ${BART_CHECKPOINT} --input_path input/valid.direct --output_path output/baseline/b50.valid.indirect.csv --num_beams 50 --num_returns 50 ;

    python shape_predict_input.py \
        --input_path input/train.direct --baseline_output_path output/baseline/b50.train.indirect.csv \
        --ref_path input/train.indirect --predict_input_path predict_input/b50.train.indirect.csv --num_beams 50 ;
    python shape_predict_input.py \
        --input_path input/valid.direct --baseline_output_path output/baseline/b50.valid.indirect.csv \
        --ref_path input/valid.indirect --predict_input_path predict_input/b50.valid.indirect.csv --num_beams 50 ;

    CUDA_VISIBLE_DEVICES=${DEVICES} python train_predict_model.py --num_beams 50 --task D2I ;
fi

if [ ${TASK} = "I2Dw" ]; then
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
fi

if [ ${TASK} = "D2Iw" ]; then
    CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
        --checkpoint ${BART_CHECKPOINT} --input_path input/train.w.direct --output_path output/baseline/b50.train.w.indirect.csv --num_beams 50 --num_returns 50 ;
    CUDA_VISIBLE_DEVICES=${DEVICES} python beam_search.py \
        --checkpoint ${BART_CHECKPOINT} --input_path input/valid.w.direct --output_path output/baseline/b50.valid.w.indirect.csv --num_beams 50 --num_returns 50 ;

    python shape_predict_input.py \
        --input_path input/train.w.direct --baseline_output_path output/baseline/b50.train.w.indirect.csv \
        --ref_path input/train.indirect --predict_input_path predict_input/b50.train.w.indirect.csv --num_beams 50 ;
    python shape_predict_input.py \
        --input_path input/valid.w.direct --baseline_output_path output/baseline/b50.valid.w.indirect.csv \
        --ref_path input/valid.indirect --predict_input_path predict_input/b50.valid.w.indirect.csv --num_beams 50 ;

    CUDA_VISIBLE_DEVICES=${DEVICES} python train_predict_model_w.py --num_beams 50 --task D2I ;
fi
