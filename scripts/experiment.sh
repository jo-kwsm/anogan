python utils/make_csv_file.py --dataset_dir dataset/img_78_28_size/
python utils/make_configs.py --model BigGAN --max_epoch 300 --size 28 --d_lr 0.000025 --beta1 0.5 --beta2 0.999

files="./result/*"
for filepath in $files; do
    if [ -d $filepath ] ; then
        python train.py "${filepath}/config.yaml"
        python evaluate.py "${filepath}/config.yaml"
    fi
done
