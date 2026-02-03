#!/bin/bash

# Rollback script for wiki-tools refactor

echo "Rolling back to main branch..."

# Switch to main branch
git checkout main

# Pull latest changes
git pull

echo "Rollback complete. Portainer will auto-update within 5 minutes."