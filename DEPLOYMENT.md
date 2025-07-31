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
- [ ] Running with `-f docker-compose.production.yml --profile production`
- [ ] Let's Encrypt certificates are generated
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

## Monitoring

```bash
# Check service status
docker-compose -f docker-compose.production.yml ps

# View logs
docker-compose -f docker-compose.production.yml logs -f

# Monitor resource usage
docker stats
``` 