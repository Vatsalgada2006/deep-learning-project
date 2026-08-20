#!/bin/bash
echo "=== DEBUG START.SH ==="
echo "Arguments received: $@"
echo "PORT environment variable: '$PORT'"
echo "======================"

# Execute the command passed as arguments
exec "$@"