# Discord Monitoring Assistant - Production Deployment Guide

This guide covers deploying the Discord Monitoring Assistant with comprehensive error handling and 24/7 reliability features.

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.11+
- Discord user token
- Google Gemini API key (or OpenAI/OpenRouter)
- Docker (for containerized deployment)

### 2. Environment Setup

Copy the environment template:
```bash
cp .env.example .env
```

Edit `.env` with your credentials:
```bash
# Required
DISCORD_TOKEN=your_discord_token_here
GOOGLE_API_KEY=your_google_api_key_here

# Optional LLM providers
OPENAI_API_KEY=your_openai_key_here
OPENROUTER_API_KEY=your_openrouter_key_here

# Configuration
ENVIRONMENT=production
LOG_LEVEL=INFO
WORKER_COUNT=4
MEMORY_LIMIT_MB=1024
LLM_DAILY_COST_LIMIT=10.0

# Email alerts (optional)
EMAIL_ENABLED=false
EMAIL_SMTP_HOST=smtp.gmail.com
EMAIL_USERNAME=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
EMAIL_FROM=discord-monitor@company.com
EMAIL_TO=admin@company.com

# Slack alerts (optional)
SLACK_ENABLED=false
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

### 3. Docker Deployment (Recommended)

```bash
# Build and start
docker-compose up -d

# Check status
docker-compose ps
docker-compose logs discord-monitor

# Stop
docker-compose down
```

### 4. Manual Deployment

```bash
# Install dependencies
pip install -r requirements-production.txt

# Create sample configuration
python production_config.py create-sample

# Validate configuration
python production_config.py

# Start service
python enhanced_main.py
```

## 🏗️ Architecture Overview

The enhanced system includes:

### Core Components

1. **Enhanced Discord Client** (`lib/resilient_discord_client.py`)
   - Rate limiting and backoff
   - Connection health monitoring
   - Automatic reconnection
   - Request history tracking

2. **Resilient Database Layer** (`lib/resilient_database.py`)
   - Connection pooling
   - Transaction management
   - Automatic backups
   - Integrity monitoring
   - Deadlock detection

3. **LLM Integration** (`lib/resilient_llm.py`)
   - Multiple provider support
   - Automatic fallbacks
   - Cost tracking and limits
   - Token management

4. **Processing Manager** (`lib/processing_manager.py`)
   - Queue overflow protection
   - Memory leak detection
   - Deadlock resolution
   - Worker health monitoring

5. **Monitoring System** (`lib/monitoring.py`)
   - Structured logging
   - Metrics collection
   - Alert management
   - Performance profiling

### Error Handling Framework

The system uses comprehensive custom exceptions (`lib/exceptions.py`) for:
- Discord API errors (rate limits, auth, forbidden)
- LLM integration errors (token limits, costs, timeouts)
- Database errors (locks, corruption, constraints)
- Processing errors (queues, memory, deadlocks)

### Resilience Features

- **Circuit Breakers**: Prevent cascading failures
- **Retry Logic**: Exponential backoff with jitter
- **Rate Limiting**: Token bucket algorithms
- **Health Monitoring**: Component health checks
- **Graceful Degradation**: Fallback strategies

## 📊 Monitoring and Alerting

### Health Check Endpoints

The service exposes HTTP endpoints for monitoring:

```bash
# Basic health check
curl http://localhost:8080/health

# Detailed status
curl http://localhost:8080/health/detailed

# Metrics
curl http://localhost:8080/metrics

# Configuration (sanitized)
curl http://localhost:8080/config
```

### Alert Channels

Configure multiple notification channels:

1. **Email Notifications**
   - SMTP configuration
   - HTML formatted alerts
   - Severity-based filtering

2. **Slack Integration**
   - Webhook notifications
   - Rich formatting
   - Channel routing

3. **Generic Webhooks**
   - Custom webhook endpoints
   - JSON payload
   - Custom headers

### Default Alerts

Pre-configured alerts for:
- High memory usage (>85%)
- High CPU usage (>90%)
- Discord API error spikes
- Database integrity issues
- Processing pipeline failures

## 🔧 Configuration

### Production Configuration File

Create `production_config.json`:

```json
{
  "environment": "production",
  "debug_mode": false,
  "discord": {
    "token": "your_token",
    "rate_limit_requests_per_minute": 50,
    "max_retries": 3,
    "connection_timeout": 30.0
  },
  "llm": {
    "google_api_key": "your_key",
    "daily_cost_limit": 10.0,
    "default_provider": "gemini",
    "fallback_providers": ["gemini-1.5-flash", "gemini-2.5-pro"]
  },
  "database": {
    "path": "data/db.sqlite",
    "backup_interval_hours": 6,
    "max_backups": 48,
    "connection_pool_size": 10
  },
  "processing": {
    "max_queue_size": 10000,
    "worker_count": 4,
    "memory_limit_mb": 1024.0
  },
  "monitoring": {
    "log_level": "INFO",
    "log_dir": "logs",
    "health_check_interval_seconds": 30
  },
  "alerts": {
    "email_enabled": true,
    "email_smtp_host": "smtp.gmail.com",
    "email_smtp_port": 587,
    "email_username": "alerts@company.com",
    "email_password": "app_password",
    "email_from": "discord-monitor@company.com",
    "email_to": ["admin@company.com"]
  }
}
```

### Environment Variables

All configuration can be overridden with environment variables:

```bash
# Discord
DISCORD_TOKEN=your_token
DISCORD_RATE_LIMIT_RPM=50

# LLM
GOOGLE_API_KEY=your_key
LLM_DAILY_COST_LIMIT=10.0

# Database
DATABASE_PATH=data/db.sqlite
DATABASE_BACKUP_INTERVAL_HOURS=6

# Processing
WORKER_COUNT=4
MAX_QUEUE_SIZE=10000
MEMORY_LIMIT_MB=1024

# Monitoring
LOG_LEVEL=INFO
HEALTH_CHECK_PORT=8080

# Alerts
EMAIL_ENABLED=true
EMAIL_SMTP_HOST=smtp.gmail.com
SLACK_ENABLED=true
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
```

## 🐳 Docker Configuration

### Resource Limits

```yaml
deploy:
  resources:
    limits:
      memory: 1.5G
      cpus: '2'
    reservations:
      memory: 512M
      cpus: '0.5'
```

### Volumes

```yaml
volumes:
  - discord_data:/app/data          # Database and persistent data
  - discord_logs:/app/logs          # Application logs
  - discord_backups:/app/backups    # Database backups
```

### Health Checks

```yaml
healthcheck:
  test: ["CMD", "/app/healthcheck.sh"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s
```

## 🛠️ Maintenance

### Database Management

```bash
# Check database statistics
python scripts/manage_db.py --stats

# Create backup
python scripts/manage_db.py --backup

# Restore from backup
python scripts/manage_db.py --restore backup_file.sqlite

# Vacuum and optimize
python scripts/manage_db.py --vacuum
```

### Log Management

Logs are automatically rotated:
- **Application logs**: `logs/discord_monitor.jsonl` (100MB, 5 files)
- **Error logs**: `logs/errors.jsonl` (50MB, 3 files)
- **JSON format**: Structured for analysis

### Backup Strategy

- **Database backups**: Every 6 hours (configurable)
- **Retention**: 48 backups (2 days)
- **Location**: `data/backups/` directory
- **Format**: SQLite with metadata

## 🚨 Troubleshooting

### Common Issues

1. **Discord Token Invalid**
   ```bash
   # Check logs
   docker-compose logs discord-monitor | grep "401\|auth"
   
   # Verify token
   curl -H "Authorization: YOUR_TOKEN" https://discord.com/api/v9/users/@me
   ```

2. **Memory Issues**
   ```bash
   # Check memory usage
   curl http://localhost:8080/metrics | grep memory
   
   # Adjust limits
   export MEMORY_LIMIT_MB=2048
   docker-compose restart discord-monitor
   ```

3. **Database Locked**
   ```bash
   # Check database health
   curl http://localhost:8080/health/detailed
   
   # Manual recovery
   python scripts/manage_db.py --repair
   ```

4. **Rate Limiting**
   ```bash
   # Check rate limit status
   curl http://localhost:8080/status | jq '.discord_client_health'
   
   # Adjust rate limits
   export DISCORD_RATE_LIMIT_RPM=30
   ```

### Log Analysis

```bash
# Error patterns
grep "ERROR" logs/discord_monitor.jsonl | jq '.exception'

# Performance metrics
grep "performance" logs/discord_monitor.jsonl | jq '.message'

# Discord API issues
grep "discord.api" logs/discord_monitor.jsonl | jq '.context'

# Database issues
grep "database" logs/errors.jsonl | jq '.error_type'
```

### Health Check Debugging

```bash
# Service health
curl -v http://localhost:8080/health

# Detailed component status
curl http://localhost:8080/health/detailed | jq '.health.components'

# Processing pipeline status
curl http://localhost:8080/status | jq '.processing_status'

# Recent metrics
curl http://localhost:8080/metrics | jq '.metrics | keys'
```

## 📈 Performance Optimization

### Resource Tuning

1. **Worker Count**: Scale with CPU cores
   ```bash
   WORKER_COUNT=8  # For 8-core systems
   ```

2. **Memory Limits**: Adjust for workload
   ```bash
   MEMORY_LIMIT_MB=2048  # For large servers
   ```

3. **Queue Sizes**: Balance latency vs memory
   ```bash
   MAX_QUEUE_SIZE=20000  # For high-volume environments
   ```

### Database Optimization

1. **Connection Pool**: Increase for high concurrency
   ```json
   "database": {
     "connection_pool_size": 20
   }
   ```

2. **Backup Frequency**: Reduce for better performance
   ```json
   "database": {
     "backup_interval_hours": 12
   }
   ```

### Rate Limiting

1. **Discord API**: Conservative limits
   ```json
   "discord": {
     "rate_limit_requests_per_minute": 40
   }
   ```

2. **LLM APIs**: Balance cost vs performance
   ```json
   "llm": {
     "daily_cost_limit": 25.0
   }
   ```

## 🔐 Security

### Token Management

- Store tokens in environment variables
- Use Docker secrets in production
- Rotate tokens regularly
- Monitor for unauthorized usage

### Network Security

- Expose only health check port (8080)
- Use reverse proxy for HTTPS
- Implement IP whitelisting
- Monitor network traffic

### Data Protection

- Encrypt sensitive data at rest
- Use secure backup storage
- Implement data retention policies
- Monitor access patterns

## 📋 Production Checklist

### Pre-Deployment

- [ ] Configuration validated
- [ ] Credentials secured
- [ ] Health checks configured
- [ ] Monitoring alerts set up
- [ ] Backup strategy in place
- [ ] Resource limits defined
- [ ] Network security configured

### Post-Deployment

- [ ] Service health verified
- [ ] Logs collection working
- [ ] Metrics being collected
- [ ] Alerts firing correctly
- [ ] Database backups created
- [ ] Performance baselines established
- [ ] Documentation updated

### Ongoing Maintenance

- [ ] Monitor resource usage
- [ ] Review error logs weekly
- [ ] Rotate credentials quarterly
- [ ] Update dependencies monthly
- [ ] Test backup recovery
- [ ] Performance optimization
- [ ] Security audit annually

For additional support, check the logs, health endpoints, and monitoring dashboards. The system is designed for self-healing, but manual intervention may be needed for extreme scenarios.