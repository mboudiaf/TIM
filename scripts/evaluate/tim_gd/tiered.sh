
# ===========================> Resnet 18 <=========================================

python3 -m src.main \
		-F logs/tim_gd/tiered/resnet18 \
		with dataset.path="data/tiered_imagenet/data" \
		ckpt_path="checkpoints/tiered/softmax/resnet18" \
		dataset.split_dir="split/tiered" \
		model.arch='resnet18' \
		model.num_classes=351 \
		evaluate=True \
		tim.iter=1000 \
		eval_parallel.method='tim_gd' \

# ===========================> WRN 28-10 <=========================================

python3 -m src.main \
		-F logs/tim_gd/tiered/wideres \
		with dataset.path="data/tiered_imagenet/data" \
		ckpt_path="checkpoints/tiered/softmax/wideres" \
		dataset.split_dir="split/tiered" \
		model.arch='wideres' \
		model.num_classes=351 \
		tim.iter=1000 \
		evaluate=True \
		eval_parallel.method='tim_gd' \

# ===========================> DenseNet <=========================================

python3 -m src.main \
		-F logs/tim_gd/tiered/densenet121 \
		with dataset.path="data/tiered_imagenet/data" \
		ckpt_path="checkpoints/tiered/softmax/densenet121" \
		dataset.split_dir="split/tiered" \
		dataset.batch_size=16 \
		model.arch='densenet121' \
		model.num_classes=351 \
		tim.iter=1000 \
		evaluate=True \
		eval_parallel.method='tim_gd' \

