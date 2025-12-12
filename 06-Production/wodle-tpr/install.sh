#!/bin/bash

set -e

INSTALL_DIR="/var/ossec/wodles/wodle-tpr"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Installing wodle-tpr to ${INSTALL_DIR}..."

if [ ! -d "/var/ossec" ]; then
    echo "Error: /var/ossec directory not found. Is Wazuh installed?"
    exit 1
fi

sudo mkdir -p "${INSTALL_DIR}"

echo "Copying files..."
sudo cp -r "${SCRIPT_DIR}"/* "${INSTALL_DIR}/"

echo "Installing Python dependencies..."
sudo pip3 install -r "${INSTALL_DIR}/requirements.txt"

echo "Setting up environment..."
if [ ! -f "${INSTALL_DIR}/.env" ]; then
    sudo cp "${INSTALL_DIR}/.env.example" "${INSTALL_DIR}/.env"
    echo "Created .env file. Please edit ${INSTALL_DIR}/.env with your OpenSearch credentials."
fi

echo "Setting permissions..."
sudo chown -R ossec:ossec "${INSTALL_DIR}"
sudo chmod 755 "${INSTALL_DIR}"
sudo chmod 755 "${INSTALL_DIR}/main.py"
sudo chmod 644 "${INSTALL_DIR}/config.json"
sudo chmod 600 "${INSTALL_DIR}/.env"

echo ""
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Edit ${INSTALL_DIR}/.env with your OpenSearch credentials"
echo "2. Copy trained models to ${INSTALL_DIR}/models/"
echo "3. Add the following to /var/ossec/etc/ossec.conf:"
echo ""
echo "<wodle name=\"command\">"
echo "  <disabled>no</disabled>"
echo "  <tag>wodle-tpr</tag>"
echo "  <command>${INSTALL_DIR}/main.py</command>"
echo "  <interval>1m</interval>"
echo "  <run_on_start>yes</run_on_start>"
echo "  <timeout>300</timeout>"
echo "</wodle>"
echo ""
echo "4. Restart Wazuh Manager: systemctl restart wazuh-manager"
echo "5. Monitor logs: tail -f /var/ossec/logs/anomaly_detection.log"
