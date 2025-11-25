#!/bin/bash

# Настройки - убедись что lab4 это правильное имя проекта
IMAGE_NAME="harbor.k8s.labs.itmo.loc/lab4/ml-inference"
TAG="latest"

echo "Building Docker image..."
docker build -t $IMAGE_NAME:$TAG .

echo "Logging in to Harbor registry..."
docker login harbor.k8s.labs.itmo.loc

echo "Pushing image to registry..."
docker push $IMAGE_NAME:$TAG

echo "Build and push completed!"