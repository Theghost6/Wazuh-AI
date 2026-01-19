#default
!228 ar.conf
restart-ossec0 - restart-ossec.sh - 0
restart-ossec0 - restart-ossec.cmd - 0
restart-wazuh0 - restart-ossec.sh - 0
restart-wazuh0 - restart-ossec.cmd - 0
restart-wazuh0 - restart-wazuh - 0
restart-wazuh0 - restart-wazuh.exe - 0
!146 agent.conf
<agent_config>
  <localfile>
    <log_format>syslog</log_format>
    <location>/home/MRs/test.log</location>
  </localfile>
</agent_config>
