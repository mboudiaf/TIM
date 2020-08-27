
# ===========================> Resnet 18 <=========================================

python3 -m src.main \
		-F logs/tim_gd/mini2cub/resnet18 \
		with dataset.path="data/mini_imagenet" \
		ckpt_path="checkpoints/mini2cub/softmax/resnet18" \
		dataset.split_dir="split/mini" \
		model.arch='resnet18' \
		tim.iter=1000 \
		evaluate=True \
		eval_parallel.method='tim_gd' \
		eval_parallel.target_data_path="data/cub/CUB_200_2011/images" \
		eval_parallel.target_split_dir="split/cub"