DEVICES=$1          # Specify GPU device to use
BART_CHECKPOINT=$2  # Specify checkpoints for BART models trained on the direct corpus
TASK=$3             # Specify a task from "I2D", "D2I", "I2Dw", "D2Iw"
                    # - I2D: Indirect-to-Direct w/o history task
                    # - D2I: Direct-to-Indirect w/o history task
                    # - I2Dw: Indirect-to-Direct w/ history task
                    # - D2Iw: Direct-to-Indirect w/ history task

mkdir -p output/both ;
mkdir -p output/positive ;
mkdir -p output/negative ;
mkdir -p output/_both ;
mkdir -p output/_positive ;
mkdir -p output/_negative ;

if [ ${TASK} = "I2D" ]; then
    CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
        --input_path input/test.indirect \
        --output_file output/both/b40.majority.s1.a0.5.n10.p4.b4.test.direct \
        --constraint constraint/both/b40.majority.test.direct.pkl \
        --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
        --length_penalty 0.6 \
        --prune_factor 4 --sat_tolerance 10 --beta 0.25 \
        --look_ahead_step 1 --alpha 0.5 --look_ahead_width 1 ;

    CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
        --input_path input/test.indirect \
        --output_file output/positive/b40.majority.s1.a0.01.n3.p4.b4.test.direct \
        --constraint constraint/positive/b40.majority.test.direct.pkl \
        --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
        --length_penalty 0.6 \
        --prune_factor 4 --sat_tolerance 3 --beta 0.25 \
        --look_ahead_step 1 --alpha 0.01 --look_ahead_width 1 ;

    CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
        --input_path input/test.indirect \
        --output_file output/negative/b60.majority.s1.a0.01.n3.p1.b4.test.direct \
        --constraint constraint/negative/b60.majority.test.direct.pkl \
        --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
        --length_penalty 0.6 \
        --prune_factor 1 --sat_tolerance 3 --beta 0.25 \
        --look_ahead_step 1 --alpha 0.01 --look_ahead_width 1 ;


    CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
        --input_path input/test.indirect \
        --output_file output/_both/b100.majority.s1.a0.02.n1.p100.b4.test.direct \
        --constraint constraint/_both/b100.majority.test.direct.pkl \
        --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
        --length_penalty 0.6 \
        --prune_factor 100 --sat_tolerance 1 --beta 0.25 \
        --look_ahead_step 1 --alpha 0.02 --look_ahead_width 1 ;

    CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
        --input_path input/test.indirect \
        --output_file output/_positive/b100.majority.s1.a0.05.n1.p20.b4.test.direct \
        --constraint constraint/_positive/b100.majority.test.direct.pkl \
        --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
        --length_penalty 0.6 \
        --prune_factor 20 --sat_tolerance 1 --beta 0.25 \
        --look_ahead_step 1 --alpha 0.05 --look_ahead_width 1 ;

    CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
        --input_path input/test.indirect \
        --output_file output/_negative/b100.majority.s1.a0.05.n1.p40.b4.test.direct \
        --constraint constraint/_negative/b100.majority.test.direct.pkl \
        --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
        --length_penalty 0.6 \
        --prune_factor 40 --sat_tolerance 1 --beta 0.25 \
        --look_ahead_step 1 --alpha 0.05 --look_ahead_width 1 ;
fi

if [ ${TASK} = "D2I" ]; then
    CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
        --input_path input/test.direct \
        --output_file output/both/b30.majority.s1.a0.5.n10.p4.b4.test.indirect \
        --constraint constraint/both/b30.majority.test.indirect.pkl \
        --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
        --length_penalty 0.6 \
        --prune_factor 4 --sat_tolerance 10 --beta 0.25 \
        --look_ahead_step 1 --alpha 0.5 --look_ahead_width 1 ;

    CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
        --input_path input/test.direct \
        --output_file output/positive/b70.majority.s1.a0.01.n3.p4.b4.test.indirect \
        --constraint constraint/positive/b70.majority.test.indirect.pkl \
        --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
        --length_penalty 0.6 \
        --prune_factor 4 --sat_tolerance 3 --beta 0.25 \
        --look_ahead_step 1 --alpha 0.01 --look_ahead_width 1 ;

    CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
        --input_path input/test.direct \
        --output_file output/negative/b10.majority.s1.a0.01.n5.p1.b4.test.indirect \
        --constraint constraint/negative/b10.majority.test.indirect.pkl \
        --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
        --length_penalty 0.6 \
        --prune_factor 1 --sat_tolerance 5 --beta 0.25 \
        --look_ahead_step 1 --alpha 0.01 --look_ahead_width 1 ;


    CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
        --input_path input/test.direct \
        --output_file output/_both/b100.majority.s1.a0.03.n1.p200.b4.test.indirect \
        --constraint constraint/_both/b100.majority.test.indirect.pkl \
        --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
        --length_penalty 0.6 \
        --prune_factor 200 --sat_tolerance 1 --beta 0.25 \
        --look_ahead_step 1 --alpha 0.03 --look_ahead_width 1 ;

    CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
        --input_path input/test.direct \
        --output_file output/_positive/b100.majority.s1.a0.05.n1.p30.b4.test.indirect \
        --constraint constraint/_positive/b100.majority.test.indirect.pkl \
        --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
        --length_penalty 0.6 \
        --prune_factor 30 --sat_tolerance 1 --beta 0.25 \
        --look_ahead_step 1 --alpha 0.05 --look_ahead_width 1 ;

    CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
        --input_path input/test.direct \
        --output_file output/_negative/b100.majority.s1.a0.05.n1.p40.b4.test.indirect \
        --constraint constraint/_negative/b100.majority.test.indirect.pkl \
        --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
        --length_penalty 0.6 \
        --prune_factor 40 --sat_tolerance 1 --beta 0.25 \
        --look_ahead_step 1 --alpha 0.05 --look_ahead_width 1 ;
fi

if [ ${TASK} = "I2Dw" ]; then
    CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
        --input_path input/test.w.indirect \
        --output_file output/both/b50.majority.s1.a0.5.n10.p4.b4.test.w.direct \
        --constraint constraint/both/b50.majority.test.w.direct.pkl \
        --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
        --length_penalty 0.6 \
        --prune_factor 4 --sat_tolerance 10 --beta 0.25 \
        --look_ahead_step 1 --alpha 0.5 --look_ahead_width 1 ;

    CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
        --input_path input/test.w.indirect \
        --output_file output/positive/b50.majority.s1.a0.01.n3.p4.b4.test.w.direct \
        --constraint constraint/positive/b50.majority.test.w.direct.pkl \
        --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
        --length_penalty 0.6 \
        --prune_factor 4 --sat_tolerance 3 --beta 0.25 \
        --look_ahead_step 1 --alpha 0.01 --look_ahead_width 1 ;

    CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
        --input_path input/test.w.indirect \
        --output_file output/negative/b50.majority.s1.a0.01.n5.p1.b4.test.w.direct \
        --constraint constraint/negative/b50.majority.test.w.direct.pkl \
        --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
        --length_penalty 0.6 \
        --prune_factor 1 --sat_tolerance 5 --beta 0.25 \
        --look_ahead_step 1 --alpha 0.01 --look_ahead_width 1 ;


    CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
        --input_path input/test.w.indirect \
        --output_file output/_both/b100.majority.s1.a0.02.n1.p200.b4.test.w.direct \
        --constraint constraint/_both/b100.majority.test.w.direct.pkl \
        --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
        --length_penalty 0.6 \
        --prune_factor 200 --sat_tolerance 1 --beta 0.25 \
        --look_ahead_step 1 --alpha 0.02 --look_ahead_width 1 ;

    CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
        --input_path input/test.w.indirect \
        --output_file output/_positive/b100.majority.s1.a0.05.n1.p20.b4.test.w.direct \
        --constraint constraint/_positive/b100.majority.test.w.direct.pkl \
        --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
        --length_penalty 0.6 \
        --prune_factor 20 --sat_tolerance 1 --beta 0.25 \
        --look_ahead_step 1 --alpha 0.05 --look_ahead_width 1 ;

    CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
        --input_path input/test.w.indirect \
        --output_file output/_negative/b100.majority.s1.a0.05.n1.p50.b4.test.w.direct \
        --constraint constraint/_negative/b100.majority.test.w.direct.pkl \
        --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
        --length_penalty 0.6 \
        --prune_factor 50 --sat_tolerance 1 --beta 0.25 \
        --look_ahead_step 1 --alpha 0.05 --look_ahead_width 1 ;

fi

if [ ${TASK} = "D2Iw" ]; then
    CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
        --input_path input/test.w.direct \
        --output_file output/both/b20.majority.s1.a0.5.n10.p4.b4.test.w.indirect \
        --constraint constraint/both/b20.majority.test.w.indirect.pkl \
        --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
        --length_penalty 0.6 \
        --prune_factor 4 --sat_tolerance 10 --beta 0.25 \
        --look_ahead_step 1 --alpha 0.5 --look_ahead_width 1 ;

    CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
        --input_path input/test.w.direct \
        --output_file output/positive/b20.majority.s1.a0.01.n3.p4.b4.test.w.indirect \
        --constraint constraint/positive/b20.majority.test.w.indirect.pkl \
        --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
        --length_penalty 0.6 \
        --prune_factor 4 --sat_tolerance 3 --beta 0.25 \
        --look_ahead_step 1 --alpha 0.01 --look_ahead_width 1 ;

    CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
        --input_path input/test.w.direct \
        --output_file output/negative/b20.majority.s1.a0.01.n5.p1.b4.test.w.indirect \
        --constraint constraint/negative/b20.majority.test.w.indirect.pkl \
        --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
        --length_penalty 0.6 \
        --prune_factor 1 --sat_tolerance 5 --beta 0.25 \
        --look_ahead_step 1 --alpha 0.01 --look_ahead_width 1 ;


    CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
        --input_path input/test.w.direct \
        --output_file output/_both/b100.majority.s1.a0.01.n1.p100.b4.test.w.indirect \
        --constraint constraint/_both/b100.majority.test.w.indirect.pkl \
        --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
        --length_penalty 0.6 \
        --prune_factor 100 --sat_tolerance 1 --beta 0.25 \
        --look_ahead_step 1 --alpha 0.01 --look_ahead_width 1 ;

    CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
        --input_path input/test.w.direct \
        --output_file output/_positive/b100.majority.s1.a0.05.n1.p30.b4.test.w.indirect \
        --constraint constraint/_positive/b100.majority.test.w.indirect.pkl \
        --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
        --length_penalty 0.6 \
        --prune_factor 30 --sat_tolerance 1 --beta 0.25 \
        --look_ahead_step 1 --alpha 0.05 --look_ahead_width 1 ;

    CUDA_VISIBLE_DEVICES=${DEVICES} python decode.py --model_name ${BART_CHECKPOINT} \
        --input_path input/test.w.direct \
        --output_file output/_negative/b100.majority.s1.a0.05.n1.p40.b4.test.w.indirect \
        --constraint constraint/_negative/b100.majority.test.w.indirect.pkl \
        --batch_size 1 --beam_size 4 --max_tgt_length 1024 --min_tgt_length 3 \
        --length_penalty 0.6 \
        --prune_factor 40 --sat_tolerance 1 --beta 0.25 \
        --look_ahead_step 1 --alpha 0.05 --look_ahead_width 1 ;
fi
