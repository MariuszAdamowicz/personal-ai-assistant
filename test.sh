#!/bin/bash

docker buildx build -t bot .
docker run --env-file .env bot
