# # ===========================> TIM-GD <=========================================

python3 -m src.main \
		-F logs/tim_gd/mini_10_ways/resnet18 \
		with dataset.path="data/mini_imagenet" \
		tim.iter=1000 \
		ckpt_path="checkpoints/mini/softmax/resnet18" \
		dataset.split_dir="split/mini" \
		model.arch='resnet18' \
		evaluate=True \
		eval_parallel.method="tim_gd" \
		eval_parallel.meta_val_way=10 \
		eval_parallel.meta_test_iter=1000

python3 -m src.main \
		-F logs/tim_gd/mini_20_ways/resnet18 \
		with dataset.path="data/mini_imagenet" \
		tim.iter=1000 \
		ckpt_path="checkpoints/mini/softmax/resnet18" \
		dataset.split_dir="split/mini" \
		model.arch='resnet18' \
		evaluate=True \
		eval_parallel.method="tim_gd" \
		eval_parallel.meta_val_way=20 \
		eval_parallel.meta_test_iter=1000
