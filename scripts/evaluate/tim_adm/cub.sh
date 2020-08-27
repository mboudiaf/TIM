# ===========================> Resnet18 <=========================================

python3 -m src.main \
		-F logs/tim_adm/cub/resnet18 \
		with dataset.path="data/cub/CUB_200_2011/images" \
		ckpt_path="checkpoints/cub/softmax/resnet18" \
		dataset.split_dir="split/cub" \
		evaluate=True \
		model.arch='resnet18' \
		model.num_classes=100 \
		eval.method="tim_adm" \

