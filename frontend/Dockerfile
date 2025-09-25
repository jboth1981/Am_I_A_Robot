# Multi-stage build for environment-aware deployment
FROM node:18-alpine AS base

WORKDIR /app
COPY package*.json ./
RUN npm ci

COPY . .

# Development stage
FROM base AS development
EXPOSE 3000
CMD ["npm", "start"]

# Production build stage
FROM base AS build
RUN npm run build

# Production serve stage
FROM nginx:alpine AS production
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]

# Default stage - will be overridden by target
FROM development