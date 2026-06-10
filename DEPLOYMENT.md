# Production Deployment Guide

## Security First: What NOT to Commit

**Never commit these files to GitHub:**
- `.env` files (contains email and environment settings)
- `certs/` directory (contains private SSL keys)
- Any API keys or secrets

## Deployment Options

### Option 1: GitHub + Server Setup (Recommended)

#### 1. Prepare Your Repository
```bash
# Make sure .gitignore is in place (already done)
git add .
git commit -m "Production ready code"
git push origin main
```

#### 2. Set Up Production Server
```bash
# Clone the repository
git clone https://github.com/yourusername/Am_I_A_Robot.git
cd Am_I_A_Robot

# Create production environment file
echo "ENV=production" > .env
echo "DEFAULT_EMAIL=your-email@example.com" >> .env
# REQUIRED: a strong random JWT signing secret. If this is missing the backend
# falls back to a random ephemeral key (sessions reset on every restart).
echo "SECRET_KEY=$(python3 -c 'import secrets; print(secrets.token_urlsafe(48))')" >> .env

# Ensure your domains point to this server's IP
# amiarobot.ca, www.amiarobot.ca, api.amiarobot.ca → Your Server IP
```

#### 3. Deploy
```bash
# Start with production profile (enables Let's Encrypt)
docker-compose -f docker-compose.production.yml --profile production up --build -d

# Check logs
docker-compose -f docker-compose.production.yml logs -f
```

### Option 2: Manual Transfer

#### 1. Transfer Files to Server
```bash
# Copy files (excluding sensitive ones)
scp -r backend/ user@your-server:/path/to/app/
scp -r frontend/ user@your-server:/path/to/app/
scp docker-compose.production.yml user@your-server:/path/to/app/
scp nginx/default.conf user@your-server:/path/to/app/nginx/
scp README.md user@your-server:/path/to/app/
scp README_LOCAL.md user@your-server:/path/to/app/
```

#### 2. Set Up on Server
```bash
# Create .env file
echo "ENV=production" > .env

# Start deployment
docker-compose -f docker-compose.production.yml --profile production up --build -d
```

## Production Checklist

- [ ] DNS records point to your server
- [ ] Ports 80 and 443 are open
- [ ] `.env` file contains `ENV=production`
- [ ] `.env` file contains a strong `SECRET_KEY` (JWT signing secret)
- [ ] Running with `-f docker-compose.production.yml --profile production`
- [ ] Let's Encrypt certificates are generated
- [ ] Backend dependencies include `bcrypt==4.3.0` (prevents 500 errors)
- [ ] Nightly database backup cron is installed (see Database Backups)

## Important Notes

- **bcrypt Version**: The backend uses `bcrypt==4.3.0` to prevent compatibility issues with passlib. Do not upgrade to bcrypt 5.0.0+ as it has breaking changes.
- **Database**: PostgreSQL is used for user data and submission tracking
- **SSL**: Let's Encrypt certificates are automatically generated for production domains
- [ ] Frontend accessible at https://amiarobot.ca
- [ ] Backend accessible at https://api.amiarobot.ca

## Troubleshooting

### Let's Encrypt Issues
```bash
# Check ACME logs
docker-compose -f docker-compose.production.yml logs nginx-proxy-acme

# Restart ACME service
docker-compose -f docker-compose.production.yml restart nginx-proxy-acme
```

### Certificate Issues
```bash
# Check certificate status
docker-compose -f docker-compose.production.yml exec nginx-proxy ls -la /etc/nginx/certs/

# Force certificate renewal
docker-compose -f docker-compose.production.yml exec nginx-proxy-acme acme.sh --renew-all
```

## Security Notes

- **Never commit `.env` or `certs/` to Git**
- **Use strong passwords for server access**
- **Keep your server updated**
- **Monitor logs regularly**
- **Backup your data**

## Database Backups

The production Postgres data lives in the `postgres_prod_data` Docker volume on a
single droplet. `scripts/pg_backup.sh` dumps and gzips the database to
`/home/deploy/backups`, keeping the most recent 14 daily backups.

Install the nightly cron (runs at 03:30 server time):
```bash
( crontab -l 2>/dev/null; \
  echo "30 3 * * * /home/deploy/Projects/Am_I_A_Robot/scripts/pg_backup.sh >> /var/log/amiarobot-backup.log 2>&1" \
) | crontab -
```

Restore from a backup:
```bash
gunzip -c /home/deploy/backups/amiarobot_YYYYMMDD_HHMMSS.sql.gz \
  | docker exec -i postgres_production psql -U robot_user -d am_i_a_robot_prod
```

## Monitoring

```bash
# Check service status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f

# Monitor resource usage
docker stats
``` 