#!/usr/bin/env bash
# Exit instantly if any underlying command fails
set -e

echo "   ∟ 🛑 Tearing down active Docker containers and freeing sockets..."
sudo docker compose down

echo "   ∟ 🏗️ Building and spinning up immutable container planes..."
sudo docker compose up -d --build --force-recreate

echo "   ∟ ⚡ Forcing Nginx to remount local upstream socket descriptors..."
sudo systemctl restart nginx

echo "   ∟ 🟢 Master Enterprise Cluster is 100% synchronized and breathing."
