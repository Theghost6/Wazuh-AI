#!/usr/bin/env sh
set -e

# Fix permissions on custom integration script (ignore errors on Windows mounts)
if [ -f /var/ossec/integrations/custom-ai ]; then
  chmod 550 /var/ossec/integrations/custom-ai || true
  chown root:wazuh /var/ossec/integrations/custom-ai || true
fi
