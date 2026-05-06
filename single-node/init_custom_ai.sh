#!/bin/bash

# 1. Copy script safely
cp /tmp/custom-ai-src /var/ossec/integrations/custom-ai

# 2. Fix CRLF (Windows line endings) - Using sed to be safe
sed -i 's/\r//g' /var/ossec/integrations/custom-ai

# 3. Set Permissions
chmod 750 /var/ossec/integrations/custom-ai
chown root:wazuh /var/ossec/integrations/custom-ai

# 4. Apply ossec.conf from mount
if [ -f /wazuh-config-mount/etc/ossec.conf ]; then
    cp /wazuh-config-mount/etc/ossec.conf /var/ossec/etc/ossec.conf
    # Fix CRLF for the config too
    sed -i 's/\r//g' /var/ossec/etc/ossec.conf
fi

# 5. Fix local_rules.xml CRLF just in case
sed -i 's/\r$//' /var/ossec/etc/rules/local_rules.xml 2>/dev/null || true

# 5. Create log dir if missing
mkdir -p /home/MRs && touch /home/MRs/test.log

# Execute original entrypoint
exec /init
