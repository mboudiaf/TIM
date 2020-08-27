
# ===========================> Resnet 18 <=======================================
python3 -m src.main \
		-F logs/tim_adm/mini2cub/resnet18 \
		with dataset.path="data/mini_imagenet" \
		ckpt_path="checkpoints/mini2cub/softmax/resnet18" \
		dataset.split_dir="split/mini" \
		model.arch='resnet18' \
		evaluate=True \
		eval.method="tim_adm" \
		eval.target_data_path="data/cub/CUB_200_2011/images" \
		eval.target_split_dir="split/cub"
