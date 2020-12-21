# ===============> Mini <===================

python3 -m src.main \
		with dataset.path="data/mini_imagenet" \
		visdom_port=8097 \
		dataset.split_dir="split/mini" \
		ckpt_path="checkpoints/mini/softmax/densenet121" \
		dataset.batch_size=256 \
		dataset.jitter=True \
		model.arch='densenet121' \
		model.num_classes=64 \
		optim.scheduler="multi_step" \
		epochs=90 \
		trainer.label_smoothing=0.1

# ===============> Tiered <===================
python3 -m src.main \
		with dataset.path="data/tiered_imagenet/data" \
		visdom_port=8097 \
		dataset.split_dir="split/tiered" \
		ckpt_path="checkpoints/tiered/softmax/densenet121" \
		dataset.batch_size=256 \
		dataset.jitter=True \
		model.arch='densenet121' \
		model.num_classes=351 \
		optim.scheduler="multi_step" \
		epochs=90 \
		trainer.label_smoothing=0.1
