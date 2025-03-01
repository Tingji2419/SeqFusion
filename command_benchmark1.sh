
for data in exchange ili weather-mean ETTh1-mean ETTh2-mean ECL-mean traffic-mean
do
for test_pred_len in 6 8 14 18 24 36 48
do
python main.py --is_training 0 --itr 1 --search_model --seq_norm --norm_horizon --select_one --ensemble_size 3 --data $data --seq_len 36 --pred_len 12 --label_len 18 --test_pred_len $test_pred_len --train_budget 1 --save_result_path benchmark1.txt --features M --repr_model SimMTM_zoo --e_layers 1 --d_model 64 --patch_len 16 --basic_model PatchTST --model SeqFusion --gpu 0 --model_zoo c58 c60 c61 c63 c64 c65 c68 c70 c73 c76
done 
done