# Local Development with HTTPS (Windows)

## Overview

This project uses Docker Compose for local development with full HTTPS support, including hot reload and WebSocket functionality for React development.

## Switching Between Local and Production

- **Local development:**
  - Run `docker compose -f docker-compose.local.yml up -d --build`
  - Uses `nginx:alpine` with custom config and trusted certificates
  - Includes hot reload with secure WebSocket connections
- **Production:**
  - Run `docker compose -f docker-compose.production.yml --profile production up -d --build`
  - Uses production nginx configuration with Let's Encrypt certificates

---

## Quick Start

### 1. Prerequisites
- Docker Desktop running
- PowerShell 7 (recommended)
- mkcert for trusted certificates

### 2. Install mkcert (for trusted certificates)
```powershell
winget install -e --id FiloSottile.mkcert --accept-package-agreements --accept-source-agreements
```

### 3. Update your hosts file
1. Open Notepad as Administrator
2. Open `C:\Windows\System32\drivers\etc\hosts`
3. Add these lines:
   ```
   127.0.0.1 amiarobot.ca
   127.0.0.1 www.amiarobot.ca
   127.0.0.1 api.amiarobot.ca
   ```
4. Save the file

### 4. Generate Trusted Certificates
```powershell
# Navigate to project root
Set-Location -Path 'C:\Users\[USERNAME]\coding_projects\Am_I_A_Robot'

# Install local CA
mkcert -install

# Generate certificates for all domains
mkcert amiarobot.ca www.amiarobot.ca api.amiarobot.ca localhost 127.0.0.1

# Copy certificates to correct locations
Copy-Item "amiarobot.ca+4.pem" "certs\amiarobot.ca\fullchain.pem" -Force
Copy-Item "amiarobot.ca+4-key.pem" "certs\amiarobot.ca\privkey.pem" -Force
Copy-Item "amiarobot.ca+4.pem" "certs\api.amiarobot.ca\fullchain.pem" -Force
Copy-Item "amiarobot.ca+4-key.pem" "certs\api.amiarobot.ca\privkey.pem" -Force

# Clean up temporary files
Remove-Item "amiarobot.ca+4.pem"
Remove-Item "amiarobot.ca+4-key.pem"
```

### 5. Start the Application
```powershell
docker compose -f docker-compose.local.yml up -d --build
```

### 6. Access Your Application
- **Frontend:** https://amiarobot.ca (with hot reload!)
- **Backend API:** https://api.amiarobot.ca
- **Database:** localhost:5432

No browser warnings - certificates are trusted!

---

## Hot Reload & WebSocket Configuration

This project uses **CRACO** (Create React App Configuration Override) to enable secure WebSocket connections for hot reload over HTTPS.

### Key Files:
- `frontend/craco.config.js` - WebSocket configuration
- `nginx/default.conf` - WebSocket proxy configuration
- `docker-compose.local.yml` - Environment setup

### How It Works:
1. **React dev server** connects via WebSocket to nginx
2. **nginx** proxies WebSocket connections to `/ws` endpoint
3. **CRACO** forces secure WebSocket (`wss://`) connections
4. **Hot reload** works seamlessly over HTTPS

### Troubleshooting Hot Reload:
If hot reload stops working:
1. Check browser console for WebSocket errors
2. Restart frontend container: `docker compose -f docker-compose.local.yml restart frontend`
3. Hard refresh browser (Ctrl+F5)

---

## Development Workflow

### Starting Development
```powershell
# Start all services
docker compose -f docker-compose.local.yml up -d

# View logs
docker compose -f docker-compose.local.yml logs -f frontend
```

### Making Changes
- **Frontend:** Edit files in `frontend/src/` - changes appear instantly
- **Backend:** Edit files in `backend/app/` - container restarts automatically
- **nginx:** Edit `nginx/default.conf` - restart nginx: `docker compose restart nginx-proxy`

### Stopping Services
```powershell
docker compose -f docker-compose.local.yml down
```

---

## Project Structure

```
Am_I_A_Robot/
├── frontend/
│   ├── craco.config.js          # WebSocket configuration
│   ├── src/                     # React source code
│   └── package.json             # Uses CRACO instead of react-scripts
├── backend/
│   └── app/                     # FastAPI application
├── nginx/
│   └── default.conf             # nginx with WebSocket proxy
├── certs/
│   ├── amiarobot.ca/           # Frontend certificates
│   └── api.amiarobot.ca/       # API certificates
└── docker-compose.local.yml    # Local development configuration
```

---

## Common Issues & Solutions

### Certificate Issues
- **Problem:** Browser shows "Not Secure"
- **Solution:** Re-run mkcert certificate generation steps

### WebSocket Errors
- **Problem:** "Mixed Content" or WebSocket connection failed
- **Solution:** Ensure CRACO config is correct and frontend container is rebuilt

### Hot Reload Not Working
- **Problem:** Changes don't appear automatically
- **Solution:** Check WebSocket connection in browser console, restart frontend container

### Container Won't Start
- **Problem:** Port conflicts or build errors
- **Solution:** Stop all containers: `docker compose down`, then restart

---

**Your local environment now perfectly matches production with HTTPS, hot reload, and secure WebSocket connections!** 🚀 