#!/bin/bash

INSTALL_DIR="/var/ossec/wodles/anomaly-detection"
OSSEC_CONF="/var/ossec/etc/ossec.conf"

echo "Installing Wazuh Anomaly Detection Wodle..."

if [ "$EUID" -ne 0 ]; then
  echo "Please run as root"
  exit 1
fi

mkdir -p $INSTALL_DIR
cp *.py $INSTALL_DIR/
cp requirements.txt $INSTALL_DIR/
cp .env.example $INSTALL_DIR/.env

cd $INSTALL_DIR
pip3 install -r requirements.txt

chmod +x main.py
chown -R ossec:ossec $INSTALL_DIR

echo ""
echo "Installation complete!"
echo ""
echo "Next steps:"
echo "1. Configure .env file: $INSTALL_DIR/.env"
echo "2. Add wodle configuration to $OSSEC_CONF (see ossec.conf.example)"
echo "3. Restart Wazuh manager: systemctl restart wazuh-manager"
