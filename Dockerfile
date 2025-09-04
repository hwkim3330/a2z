# Multi-stage Dockerfile for A2Z TSN/FRER System

# Stage 1: Python base
FROM python:3.11-slim AS python-base
WORKDIR /app
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libssl-dev \
    libffi-dev \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 2: Node.js base
FROM node:20-slim AS node-base
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

# Stage 3: Build stage
FROM node:20-slim AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Stage 4: Final production image
FROM python:3.11-slim
WORKDIR /app

# Install Node.js
RUN apt-get update && apt-get install -y \
    curl \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Copy Python dependencies
COPY --from=python-base /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy Node dependencies
COPY --from=node-base /app/node_modules ./node_modules

# Copy built application
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/dashboard ./dashboard
COPY --from=builder /app/docs ./docs

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 a2z && chown -R a2z:a2z /app
USER a2z

# Expose ports
EXPOSE 3000 8080 8443 9090 1883 5672

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD node healthcheck.js || exit 1

# Start command
CMD ["npm", "start"]