
# ===========================> Resnet 18 <=========================================

python3 -m src.main \
		-F logs/non_augmented/baseline/tiered/resnet18 \
		with dataset.path="data/tiered_imagenet/data" \
		ckpt_path="checkpoints/tiered/softmax/resnet18" \
		dataset.split_dir="split/tiered" \
		model.arch='resnet18' \
		model.num_classes=351 \
		evaluate=True

# ===========================> WRN 28-10 <=========================================

python3 -m src.main \
		-F logs/non_augmented/baseline/tiered/wideres \
		with dataset.path="data/tiered_imagenet/data" \
		ckpt_path="checkpoints/tiered/softmax/wideres" \
		dataset.split_dir="split/tiered" \
		model.arch='wideres' \
		model.num_classes=351 \
		evaluate=True

# ===========================> DenseNet <=========================================

python3 -m src.main \
		-F logs/non_augmented/baseline/tiered/densenet121 \
		with dataset.path="data/tiered_imagenet/data" \
		ckpt_path="checkpoints/tiered/softmax/densenet121" \
		dataset.split_dir="split/tiered" \
		dataset.batch_size=16 \
		model.arch='densenet121' \
		model.num_classes=351 \
		evaluate=True