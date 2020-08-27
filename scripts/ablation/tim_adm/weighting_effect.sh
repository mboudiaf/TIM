# ================> mini-Imagenet <=================
# ========================================

python3 -m src.main \
		-F logs/ablation/tim_adm/mini \
		with dataset.path="data/mini_imagenet" \
		ckpt_path="checkpoints/mini/softmax/resnet18" \
		dataset.split_dir="split/mini" \
		model.arch='resnet18' \
		evaluate=True \
		eval.method='tim_adm' \
        tim.loss_weights="[0.1, 0., 0.]" \


# No H(Y)
python3 -m src.main \
		-F logs/ablation/tim_adm/mini \
		with dataset.path="data/mini_imagenet" \
		ckpt_path="checkpoints/mini/softmax/resnet18" \
		dataset.split_dir="split/mini" \
		model.arch='resnet18' \
		evaluate=True \
		eval.method='tim_adm' \
		tim.loss_weights="[0.1, 0., 0.1]" \


# No H(Y|X)
python3 -m src.main \
		-F logs/ablation/tim_adm/mini \
		with dataset.path="data/mini_imagenet" \
		ckpt_path="checkpoints/mini/softmax/resnet18" \
		dataset.split_dir="split/mini" \
		model.arch='resnet18' \
		evaluate=True \
		eval.method='tim_adm' \
		tim.loss_weights="[0.1, 1.0, 0.]" \

# Full loss
python3 -m src.main \
		-F logs/ablation/tim_adm/mini \
		with dataset.path="data/mini_imagenet" \
		ckpt_path="checkpoints/mini/softmax/resnet18" \
		dataset.split_dir="split/mini" \
		model.arch='resnet18' \
		evaluate=True \
		eval.method='tim_adm' \
		tim.loss_weights="[0.1, 1.0, 0.1]" \

# ================> tiered-Imagenet <=================
# ========================================


# No H(Y)
python3 -m src.main \
		-F logs/ablation/tim_adm/tiered \
		with dataset.path="data/tiered_imagenet/data" \
		ckpt_path="checkpoints/tiered/softmax/resnet18" \
		dataset.split_dir="split/tiered" \
		model.arch='resnet18' \
		model.num_classes=351 \
		evaluate=True \
		eval.method='tim_adm' \
        tim.loss_weights="[0.1, 0., 0.]" \

# No H(Y)
python3 -m src.main \
		-F logs/ablation/tim_adm/tiered \
		with dataset.path="data/tiered_imagenet/data" \
		ckpt_path="checkpoints/tiered/softmax/resnet18" \
		dataset.split_dir="split/tiered" \
		model.arch='resnet18' \
		model.num_classes=351 \
		evaluate=True \
		eval.method='tim_adm' \
		tim.loss_weights="[0.1, 0., 0.1]" \

# No H(Y|X)
python3 -m src.main \
		-F logs/ablation/tim_adm/tiered \
		with dataset.path="data/tiered_imagenet/data" \
		ckpt_path="checkpoints/tiered/softmax/resnet18" \
		dataset.split_dir="split/tiered" \
		model.num_classes=351 \
		model.arch='resnet18' \
		evaluate=True \
		eval.method='tim_adm' \
		tim.loss_weights="[0.1, 1.0, 0.]" \

# Full loss
python3 -m src.main \
		-F logs/ablation/tim_adm/tiered \
		with dataset.path="data/tiered_imagenet/data" \
		ckpt_path="checkpoints/tiered/softmax/resnet18" \
		dataset.split_dir="split/tiered" \
		model.num_classes=351 \
		model.arch='resnet18' \
		evaluate=True \
		eval.method='tim_adm' \
		tim.loss_weights="[0.1, 1.0, 0.1]" \


# ================> CUB <=================
# ========================================

# No H(Y)
python3 -m src.main \
		-F logs/ablation/tim_adm/cub \
		with dataset.path="data/cub/CUB_200_2011/images" \
		ckpt_path="checkpoints/cub/softmax/resnet18" \
		dataset.split_dir="split/cub" \
		model.arch='resnet18' \
		model.num_classes=100 \
		evaluate=True \
		eval.method='tim_adm' \
        tim.loss_weights="[0.1, 0., 0.]" \

# No H(Y)
python3 -m src.main \
		-F logs/ablation/tim_adm/cub \
		with dataset.path="data/cub/CUB_200_2011/images" \
		ckpt_path="checkpoints/cub/softmax/resnet18" \
		dataset.split_dir="split/cub" \
		model.arch='resnet18' \
		model.num_classes=100 \
		evaluate=True \
		eval.method='tim_adm' \
		tim.loss_weights="[0.1, 0., 0.1]" \

# No H(Y|X)
python3 -m src.main \
		-F logs/ablation/tim_adm/cub \
		with dataset.path="data/cub/CUB_200_2011/images" \
		ckpt_path="checkpoints/cub/softmax/resnet18" \
		dataset.split_dir="split/cub" \
		model.arch='resnet18' \
		model.num_classes=100 \
		evaluate=True \
		eval.method='tim_adm' \
		tim.loss_weights="[0.1, 1.0, 0.]" \

# Full loss
python3 -m src.main \
		-F logs/ablation/tim_adm/cub \
		with dataset.path="data/cub/CUB_200_2011/images" \
		ckpt_path="checkpoints/cub/softmax/resnet18" \
		dataset.split_dir="split/cub" \
		model.arch='resnet18' \
		model.num_classes=100 \
		evaluate=True \
		eval.method='tim_adm' \
		tim.loss_weights="[0.1, 1.0, 0.1]" \