run-ppo-8pp:
	touch /tmp/seed.json && rm /tmp/seed.json && \
	export LD_LIBRARY_PATH=$(HOME)/.mujoco/mjpro150/bin:/usr/lib/nvidia-384 && \
	rm config.py && cp config_ppo_8pp.py config.py \
	python3 run.py

run-ppo-10maze:
	touch /tmp/seed.json && rm /tmp/seed.json && \
	export LD_LIBRARY_PATH=$(HOME)/.mujoco/mjpro150/bin:/usr/lib/nvidia-384 && \
	rm config.py && cp config_ppo_10maze.py config.py \
	python3 run.py

run-learn-10-maze-model0:
	touch /tmp/seed.json && rm /tmp/seed.json && \
	export LD_LIBRARY_PATH=$(HOME)/.mujoco/mjpro150/bin:/usr/lib/nvidia-384 && \
	rm config.py && cp config_model0.py config.py \
	python3 run.py

run-learn-10-maze-model1:
	touch /tmp/seed.json && rm /tmp/seed.json && \
	export LD_LIBRARY_PATH=$(HOME)/.mujoco/mjpro150/bin:/usr/lib/nvidia-384 && \
	rm config.py && cp config_model1.py config.py \
	python3 run.py

run-learn-10-maze-model2:
	touch /tmp/seed.json && rm /tmp/seed.json && \
	export LD_LIBRARY_PATH=$(HOME)/.mujoco/mjpro150/bin:/usr/lib/nvidia-384 && \
	rm config.py && cp config_model2.py config.py \
	python3 run.py

run-learn-10-maze-model3:
	touch /tmp/seed.json && rm /tmp/seed.json && \
	export LD_LIBRARY_PATH=$(HOME)/.mujoco/mjpro150/bin:/usr/lib/nvidia-384 && \
	rm config.py && cp config_model3.py config.py \
	python3 run.py

run-10-maze-learned-models:
	touch /tmp/seed.json && rm /tmp/seed.json && \
	export LD_LIBRARY_PATH=$(HOME)/.mujoco/mjpro150/bin:/usr/lib/nvidia-384 && \
	rm config.py && cp using_learned_models_config.py config.py \
	python3 run.py

run-10-maze-hand:
	touch /tmp/seed.json && rm /tmp/seed.json && \
	export LD_LIBRARY_PATH=$(HOME)/.mujoco/mjpro150/bin:/usr/lib/nvidia-384 && \
	rm config.py && cp config_10maze_hand.py config.py \
	python3 run.py

run-8pp-hand:
	touch /tmp/seed.json && rm /tmp/seed.json && \
	export LD_LIBRARY_PATH=$(HOME)/.mujoco/mjpro150/bin:/usr/lib/nvidia-384 && \
	rm config.py && cp config_8pp_hand.py config.py \
	python3 run.py

