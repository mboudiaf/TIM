
# ===========================> Resnet 18 <=========================================

python3 -m src.main \
		-F logs/non_augmented/baseline/cub/resnet18 \
		with dataset.path="data/cub/CUB_200_2011/images" \
		ckpt_path="checkpoints/cub/softmax/resnet18" \
		dataset.split_dir="split/cub" \
		model.arch='resnet18' \
		model.num_classes=100 \
		evaluate=True