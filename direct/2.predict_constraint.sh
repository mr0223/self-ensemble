DEVICES=$1                      # Specify GPU device to use
PREDICT_MODEL_CHECKPOINT=$2     # Specify the name of the predict model trained in 1.train_predict_model.sh
TASK=$3                         # Specify a task from "I2D", "D2I", "I2Dw", "D2Iw"
                                # - I2D: Indirect-to-Direct w/o history task
                                # - D2I: Direct-to-Indirect w/o history task
                                # - I2Dw: Indirect-to-Direct w/ history task
                                # - D2Iw: Direct-to-Indirect w/ history task

mkdir -p constraint/both ;
mkdir -p constraint/positive ;
mkdir -p constraint/negative ;
mkdir -p constraint/_both ;
mkdir -p constraint/_positive ;
mkdir -p constraint/_negative ;

if [ ${TASK} = "I2D" ]; then
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
fi

if [ ${TASK} = "D2I" ]; then
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
fi

if [ ${TASK} = "I2Dw" ]; then
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
fi

if [ ${TASK} = "D2Iw" ]; then
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
fi
