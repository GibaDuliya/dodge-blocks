#!/bin/bash

set -e

ACTION=${1:-train}

case $ACTION in
    build)
        echo "Building Docker image..."
        docker compose build
        ;;
    train)
        echo "Starting training..."
        shift
        docker compose run --rm train "$@"
        ;;
    eval|evaluate)
        echo "Starting evaluation..."
        shift
        docker compose run --rm evaluate "$@"
        ;;
    jupyter|notebook)
        echo "Starting Jupyter Lab..."
        echo "Open http://localhost:8888 in your browser"
        docker compose up jupyter
        ;;
    shell|bash)
        echo "Starting interactive shell..."
        docker compose run --rm train bash
        ;;
    *)
        echo "Usage: $0 {build|train|eval|jupyter|shell} [args...]"
        echo ""
        echo "Examples:"
        echo "  $0 build                    # Build Docker image"
        echo "  $0 train                    # Run training"
        echo "  $0 train --name exp1        # Run training with args"
        echo "  $0 eval                     # Run evaluation"
        echo "  $0 jupyter                  # Start Jupyter Lab"
        echo "  $0 shell                    # Open interactive shell"
        exit 1
        ;;
esac
