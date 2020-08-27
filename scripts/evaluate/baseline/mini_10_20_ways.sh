# # ===========================> TIM-GD <=========================================

python3 -m src.main \
		-F logs/10_ways/non_augmented/tim_gd/resnet18 \
		with dataset.path="data/mini_imagenet" \
		ckpt_path="checkpoints/mini/softmax/resnet18" \
		dataset.split_dir="split/mini" \
		model.arch='resnet18' \
		evaluate=True \
		eval_parallel.meta_val_way=10

python3 -m src.main \
		-F logs/20_ways/non_augmented/tim_gd/resnet18 \
		with dataset.path="data/mini_imagenet" \
		ckpt_path="checkpoints/mini/softmax/resnet18" \
		dataset.split_dir="split/mini" \
		model.arch='resnet18' \
		evaluate=True \
		eval_parallel.meta_val_way=20
