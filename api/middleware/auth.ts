import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { createHash } from 'crypto';

interface TokenPayload {
    userId: string;
    role: string;
    permissions: string[];
    iat: number;
    exp: number;
}

interface ApiKey {
    key: string;
    name: string;
    permissions: string[];
    rateLimit: number;
    created: Date;
    lastUsed: Date;
}

export function authMiddleware(jwtSecret: string) {
    return async (req: Request, res: Response, next: NextFunction) => {
        try {
            // Check for API key authentication
            const apiKey = req.headers['x-api-key'] as string;
            if (apiKey) {
                const isValid = await validateApiKey(apiKey);
                if (isValid) {
                    req.user = {
                        type: 'api_key',
                        key: hashApiKey(apiKey),
                        permissions: await getApiKeyPermissions(apiKey)
                    };
                    return next();
                }
            }

            // Check for JWT Bearer token
            const authHeader = req.headers.authorization;
            if (!authHeader || !authHeader.startsWith('Bearer ')) {
                // Allow public endpoints
                if (isPublicEndpoint(req.path, req.method)) {
                    return next();
                }
                
                return res.status(401).json({
                    error: 'UNAUTHORIZED',
                    message: 'Authentication required'
                });
            }

            const token = authHeader.substring(7);
            const payload = jwt.verify(token, jwtSecret) as TokenPayload;

            // Check token expiration
            if (payload.exp && payload.exp < Date.now() / 1000) {
                return res.status(401).json({
                    error: 'TOKEN_EXPIRED',
                    message: 'Token has expired'
                });
            }

            // Attach user to request
            req.user = {
                type: 'jwt',
                userId: payload.userId,
                role: payload.role,
                permissions: payload.permissions
            };

            next();
        } catch (error) {
            if (error.name === 'JsonWebTokenError') {
                return res.status(401).json({
                    error: 'INVALID_TOKEN',
                    message: 'Invalid authentication token'
                });
            }
            
            return res.status(500).json({
                error: 'AUTH_ERROR',
                message: 'Authentication failed'
            });
        }
    };
}

export function requirePermission(permission: string) {
    return (req: Request, res: Response, next: NextFunction) => {
        if (!req.user) {
            return res.status(401).json({
                error: 'UNAUTHORIZED',
                message: 'Authentication required'
            });
        }

        if (!req.user.permissions || !req.user.permissions.includes(permission)) {
            return res.status(403).json({
                error: 'FORBIDDEN',
                message: `Insufficient permissions. Required: ${permission}`
            });
        }

        next();
    };
}

export function requireRole(role: string) {
    return (req: Request, res: Response, next: NextFunction) => {
        if (!req.user) {
            return res.status(401).json({
                error: 'UNAUTHORIZED',
                message: 'Authentication required'
            });
        }

        const userRole = req.user.role;
        const roleHierarchy = ['viewer', 'operator', 'engineer', 'admin', 'super_admin'];
        
        const userRoleIndex = roleHierarchy.indexOf(userRole);
        const requiredRoleIndex = roleHierarchy.indexOf(role);

        if (userRoleIndex < requiredRoleIndex) {
            return res.status(403).json({
                error: 'FORBIDDEN',
                message: `Insufficient role. Required: ${role}`
            });
        }

        next();
    };
}

async function validateApiKey(apiKey: string): Promise<boolean> {
    // In production, check against database
    const validKeys = process.env.VALID_API_KEYS?.split(',') || [];
    const hashedKey = hashApiKey(apiKey);
    return validKeys.includes(hashedKey);
}

function hashApiKey(apiKey: string): string {
    return createHash('sha256').update(apiKey).digest('hex');
}

async function getApiKeyPermissions(apiKey: string): Promise<string[]> {
    // In production, fetch from database based on API key
    // For now, return default permissions
    return [
        'read:network',
        'read:monitoring',
        'read:frer',
        'write:alerts'
    ];
}

function isPublicEndpoint(path: string, method: string): boolean {
    const publicEndpoints = [
        { path: '/health', method: 'GET' },
        { path: '/ready', method: 'GET' },
        { path: '/metrics', method: 'GET' },
        { path: '/api-docs', method: 'GET' }
    ];

    return publicEndpoints.some(endpoint => 
        path.startsWith(endpoint.path) && method === endpoint.method
    );
}

export function generateToken(userId: string, role: string, permissions: string[], secret: string): string {
    const payload: Omit<TokenPayload, 'iat' | 'exp'> = {
        userId,
        role,
        permissions
    };

    return jwt.sign(payload, secret, {
        expiresIn: '24h',
        issuer: 'a2z-tsn',
        audience: 'a2z-api'
    });
}

export function refreshToken(token: string, secret: string): string {
    try {
        const payload = jwt.verify(token, secret, {
            ignoreExpiration: true
        }) as TokenPayload;

        // Check if token is not too old (within 7 days)
        const tokenAge = Date.now() / 1000 - payload.iat;
        if (tokenAge > 7 * 24 * 60 * 60) {
            throw new Error('Token too old for refresh');
        }

        return generateToken(payload.userId, payload.role, payload.permissions, secret);
    } catch (error) {
        throw new Error('Failed to refresh token');
    }
}

export class RateLimiter {
    private requests: Map<string, number[]> = new Map();
    private readonly windowMs: number;
    private readonly maxRequests: number;

    constructor(windowMs: number = 60000, maxRequests: number = 100) {
        this.windowMs = windowMs;
        this.maxRequests = maxRequests;

        // Clean up old entries periodically
        setInterval(() => this.cleanup(), windowMs);
    }

    check(identifier: string): boolean {
        const now = Date.now();
        const requests = this.requests.get(identifier) || [];
        
        // Filter out old requests
        const recentRequests = requests.filter(time => now - time < this.windowMs);
        
        if (recentRequests.length >= this.maxRequests) {
            return false;
        }

        recentRequests.push(now);
        this.requests.set(identifier, recentRequests);
        return true;
    }

    private cleanup(): void {
        const now = Date.now();
        for (const [identifier, requests] of this.requests.entries()) {
            const recentRequests = requests.filter(time => now - time < this.windowMs);
            if (recentRequests.length === 0) {
                this.requests.delete(identifier);
            } else {
                this.requests.set(identifier, recentRequests);
            }
        }
    }
}

export const rateLimiter = new RateLimiter();

export function apiRateLimiter(req: Request, res: Response, next: NextFunction): void {
    const identifier = req.user?.key || req.ip;
    
    if (!rateLimiter.check(identifier)) {
        res.status(429).json({
            error: 'RATE_LIMIT_EXCEEDED',
            message: 'Too many requests'
        });
        return;
    }

    next();
}