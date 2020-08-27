
# ===========================> Resnet 18 <=========================================

python3 -m src.main \
		-F logs/tim_adm/tiered/resnet18 \
		with dataset.path="data/tiered_imagenet/data" \
		ckpt_path="checkpoints/tiered/softmax/resnet18" \
		dataset.split_dir="split/tiered" \
		model.arch='resnet18' \
		model.num_classes=351 \
		evaluate=True \
		eval.method="tim_adm"

# ===========================> WRN 28-10 <=========================================

python3 -m src.main \
		-F logs/tim_adm/tiered/wideres \
		with dataset.path="data/tiered_imagenet/data" \
		ckpt_path="checkpoints/tiered/softmax/wideres" \
		dataset.split_dir="split/tiered" \
		model.num_classes=351 \
		model.arch='wideres' \
		evaluate=True \
		eval.method="tim_adm"

# # ===========================> DenseNet <=========================================

python3 -m src.main \
		-F logs/tim_adm/tiered/densenet121 \
		with dataset.path="data/tiered_imagenet/data" \
		ckpt_path="checkpoints/tiered/softmax/densenet121" \
		dataset.split_dir="split/tiered" \
		model.num_classes=351 \
		model.arch='densenet121' \
		evaluate=True \
		eval.method="tim_adm"