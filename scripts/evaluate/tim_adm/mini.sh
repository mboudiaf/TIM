
# ===========================> Resnet 18 <=========================================

python3 -m src.main \
		-F logs/tim_adm/mini/resnet18 \
		with dataset.path="data/mini_imagenet" \
		ckpt_path="checkpoints/mini/softmax/resnet18" \
		dataset.split_dir="split/mini" \
		model.arch='resnet18' \
		evaluate=True \
		eval.method="tim_adm" \

# ===========================> WRN 28-10 <=========================================

python3 -m src.main \
		-F logs/tim_adm/mini/wideres \
		with dataset.path="data/mini_imagenet" \
		ckpt_path="checkpoints/mini/softmax/wideres" \
		dataset.split_dir="split/mini" \
		model.arch='wideres' \
		evaluate=True \
		eval.method="tim_adm" \

# ===========================> DenseNet <=========================================

python3 -m src.main \
		-F logs/tim_adm/mini/densenet121 \
		with dataset.path="data/mini_imagenet" \
		ckpt_path="checkpoints/mini/softmax/densenet121" \
		dataset.split_dir="split/mini" \
		model.arch='densenet121' \
		evaluate=True \
		eval.method="tim_adm" \


# python3 -m src.main with dataset.path="data/mini_imagenet" ckpt_path="checkpoints/mini/softmax/wideres" dataset.split_dir="split/mini" model.arch='wideres' evaluate=True eval.method="tim_adm"