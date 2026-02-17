.PHONY: build train eval jupyter shell clean help setup-docker

IMAGE_NAME = dodge-blocks
CONTAINER_NAME = dodge-blocks-container

# Use sudo if DOCKER_SUDO is set to 1
DOCKER_CMD = $(if $(DOCKER_SUDO),sudo ,)docker compose

help:
	@echo "Available commands:"
	@echo "  make build         - Build Docker image"
	@echo "  make train         - Run training"
	@echo "  make eval          - Run evaluation"
	@echo "  make jupyter       - Start Jupyter Lab"
	@echo "  make shell         - Open interactive shell"
	@echo "  make clean         - Clean Docker artifacts"
	@echo "  make setup-docker  - Add user to docker group (requires sudo)"
	@echo ""
	@echo "If you get permission errors, use:"
	@echo "  make build DOCKER_SUDO=1"
	@echo "  Or run: make setup-docker (then logout/login)"

setup-docker:
	@echo "Adding user to docker group..."
	@sudo usermod -aG docker $$USER
	@echo "User added to docker group."
	@echo "Please logout and login again, or run: newgrp docker"

build:
	$(DOCKER_CMD) build

train:
	$(DOCKER_CMD) run --rm train

eval:
	$(DOCKER_CMD) run --rm evaluate

jupyter:
	$(DOCKER_CMD) up jupyter

shell:
	$(DOCKER_CMD) run --rm train bash

clean:
	$(DOCKER_CMD) down
	docker rmi $(IMAGE_NAME) 2>/dev/null || true

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-train:
	docker run --gpus all --rm \
		-v $$(pwd)/artifacts:/app/artifacts \
		-v $$(pwd)/analysis:/app/analysis \
		$(IMAGE_NAME) python run/train.py

docker-eval:
	docker run --gpus all --rm \
		-v $$(pwd)/artifacts:/app/artifacts \
		-v $$(pwd)/analysis:/app/analysis \
		$(IMAGE_NAME) python run/evaluate.py --checkpoint artifacts/checkpoints/best.pt --num_episodes 100
