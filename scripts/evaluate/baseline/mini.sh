
# ===========================> Resnet 18 <=========================================

python3 -m src.main \
		-F logs/non_augmented/baseline/mini/resnet18 \
		with dataset.path="data/mini_imagenet" \
		ckpt_path="checkpoints/mini/softmax/resnet18" \
		dataset.split_dir="split/mini" \
		model.arch='resnet18' \
		evaluate=True 

# ===========================> WRN 28-10 <=========================================

python3 -m src.main \
		-F logs/non_augmented/baseline/mini/wideres \
		with dataset.path="data/mini_imagenet" \
		ckpt_path="checkpoints/mini/softmax/wideres" \
		dataset.split_dir="split/mini" \
		model.arch='wideres' \
		evaluate=True

# ===========================> DenseNet <=========================================

python3 -m src.main \
		-F logs/non_augmented/baseline/mini/densenet121 \
		with dataset.path="data/mini_imagenet" \
		ckpt_path="checkpoints/mini/softmax/densenet121" \
		dataset.split_dir="split/mini" \
		model.arch='densenet121' \
		evaluate=True \