# Local Development with HTTPS (Windows)

## Switching Between Local and Production

This project uses separate Docker Compose files for local development and production deployment.

- **Local development:**
  - Set `ENV=local` in your `.env` file.
  - Run `docker-compose -f docker-compose.local.yml up --build`.
  - Uses `nginx:alpine` with custom config and self-signed certificates.
  - No ACME companion (not needed for local development).
- **Production:**
  - Set `ENV=production` in your `.env` file.
  - Run `docker-compose -f docker-compose.production.yml --profile production up --build`.
  - Uses `jwilder/nginx-proxy` with the ACME companion for real SSL certificates.

---

## 1. Update your hosts file

1. Open Notepad as Administrator.
2. Open `C:\Windows\System32\drivers\etc\hosts`.
3. Add these lines:
   ```
   127.0.0.1 amiarobot.ca
   127.0.0.1 www.amiarobot.ca
   127.0.0.1 api.amiarobot.ca
   ```
4. Save the file.

## 2. Generate Self-Signed Certificates

Open PowerShell in your project root and run:

```
mkdir certs\amiarobot.ca
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout certs/amiarobot.ca/privkey.pem -out certs/amiarobot.ca/fullchain.pem -subj "/CN=amiarobot.ca"

mkdir certs\api.amiarobot.ca
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout certs/api.amiarobot.ca/privkey.pem -out certs/api.amiarobot.ca/fullchain.pem -subj "/CN=api.amiarobot.ca"
```

If you don't have `openssl`, install it or use Git Bash.

## 3. Start Docker Compose

From your project root:

```
docker-compose up --build
```

## 4. Access your app

- Frontend: https://amiarobot.ca
- Backend API: https://api.amiarobot.ca

You may get a browser warning about the certificate not being trusted. You can proceed past the warning for local development.

## 5. Stopping

Press `Ctrl+C` in the terminal to stop the services.

---

**Your local environment now closely matches production, including HTTPS!** 