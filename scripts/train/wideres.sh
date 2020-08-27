# ===============> Mini <===================

python3 -m src.main \
		with dataset.path="data/mini_imagenet" \
		dataset.split_dir="split/mini" \
		ckpt_path="checkpoints/mini/softmax/wideres" \
		dataset.batch_size=64 \
		dataset.jitter=True \
		model.arch='wideres' \
		model.num_classes=64 \
		optim.scheduler="multi_step" \
		epochs=90 \
		trainer.label_smoothing=0.1

# ===============> Tiered <===================

python3 -m src.main \
		with dataset.path="data/tiered_imagenet" \
		dataset.split_dir="split/tiered" \
		dataset.jitter=True \
		ckpt_path="checkpoints/tiered/softmax/wideres" \
		dataset.batch_size=256 \
		model.arch='wideres' \
		model.num_classes=64 \
		optim.scheduler="multi_step" \
		epochs=90 \
		trainer.label_smoothing=0.1

