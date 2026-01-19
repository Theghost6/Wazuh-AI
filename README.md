# Wazuh-AI
# Hướng dẫn cài đặt Wazuh-AI


## Cài đặt

### Bước 1: Clone repository
```bash
git clone https://github.com/Theghost6/Wazuh-AI.git
cd Wazuh-AI/single-node
```

### Bước 2: Tạo SSL Certificates

```bash
docker-compose -f generate-indexer-certs.yml run --rm generator
```

### Bước 3: Khởi động Wazuh
```bash
docker-compose up --build -d
```

## Truy cập

| Service | URL | Username | Password |
|---------|-----|----------|----------|
| Wazuh Dashboard | https://localhost:443 | admin | SecretPassword |
