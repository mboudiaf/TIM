
# =============> Resnet18 <================
python3 -m src.main \
		-F logs/tim_gd/cub/ \
		with dataset.path="data/cub/CUB_200_2011/images" \
		ckpt_path="checkpoints/cub/softmax/resnet18" \
		dataset.split_dir="split/cub" \
		model.arch='resnet18' \
		model.num_classes=100 \
		tim.iter=1000 \
		evaluate=True \
		eval.method='tim_gd' \

