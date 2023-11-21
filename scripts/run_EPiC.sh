model='FEDformer_EPiC'
version='EWDF'
# Where the data is
root_path=/data/EPiC_project/EPiC-2023-competition-data-v1.0/data

scenario=4
vali_fold=1

if [ $scenario -eq 1 ]
then
task_id=EPiC_sc_${scenario}
else
task_id=EPiC_sc_${scenario}_fold_${vali_fold}
fi


if [ $scenario -eq 1 ] || [ $scenario -eq 4 ] 
then
    batch_size=32
else
    batch_size=64
fi
# for single GPU running Change 'accelerate launch'
# to "python -u"
accelerate launch run.py \
  --is_training 1 \
  --root_path $root_path\
  --task_id $task_id\
  --scenario $scenario\
  --vali_fold $vali_fold\
  --model $model \
  --data EPiC \
  --features M \
  --version $version\
  --freq m\
  --seq_len 501 \
  --label_len 1 \
  --pred_len 10 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 2 \
  --c_out 2 \
  --num_workers 8\
  --learning_rate 1e-3\
  --lradj 'type5' \
  --batch_size $batch_size\
  --train_epochs 30\
  --patience 5\
  --des 'Exp' \
  --d_model 512 \
  --itr 1 

echo ">>>>>>>Extracting Ouput<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"

python -u extracted_output.py\
  --root_path $root_path\
  --task_id $task_id\
  --scenario $scenario\
  --vali_fold $vali_fold

echo ">>>>>>>Generating Smooth Prediction<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"

python -u smooth.py \
  --root_path $root_path\
  --scenario $scenario\
  --vali_fold $vali_fold\
  --label_len 1 \
  --pred_len 10 
# done

