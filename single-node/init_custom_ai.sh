#!/bin/bash

# 1. Copy script safely
cp /tmp/custom-ai-src /var/ossec/integrations/custom-ai

# 2. Fix CRLF (Windows line endings) - Using sed to be safe
sed -i 's/\r//g' /var/ossec/integrations/custom-ai

# 3. Set Permissions
chmod 750 /var/ossec/integrations/custom-ai
chown root:wazuh /var/ossec/integrations/custom-ai

# 4. Fix local_rules.xml CRLF just in case
sed -i 's/\r$//' /var/ossec/etc/rules/local_rules.xml 2>/dev/null || true

# 5. Create log dir if missing
mkdir -p /home/MRs && touch /home/MRs/test.log

# Execute original entrypoint
exec /init
