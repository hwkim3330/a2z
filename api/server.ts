import express, { Request, Response, NextFunction } from 'express';
import { Server } from 'http';
import { WebSocketServer } from 'ws';
import helmet from 'helmet';
import cors from 'cors';
import compression from 'compression';
import rateLimit from 'express-rate-limit';
import swaggerUi from 'swagger-ui-express';
import YAML from 'yamljs';
import jwt from 'jsonwebtoken';
import { MongoClient, Db } from 'mongodb';
import Redis from 'ioredis';
import { InfluxDB, Point } from '@influxdata/influxdb-client';
import winston from 'winston';
import prometheus from 'prom-client';
import path from 'path';
import { v4 as uuidv4 } from 'uuid';

// Import route handlers
import { NetworkRouter } from './routes/network';
import { FRERRouter } from './routes/frer';
import { MonitoringRouter } from './routes/monitoring';
import { MLRouter } from './routes/ml';
import { BlockchainRouter } from './routes/blockchain';
import { ConfigurationRouter } from './routes/configuration';
import { FailoverRouter } from './routes/failover';

// Import middleware
import { authMiddleware } from './middleware/auth';
import { errorHandler } from './middleware/error';
import { requestLogger } from './middleware/logging';
import { metricsMiddleware } from './middleware/metrics';

// Import services
import { NetworkService } from './services/network';
import { FRERService } from './services/frer';
import { MonitoringService } from './services/monitoring';
import { MLService } from './services/ml';
import { BlockchainService } from './services/blockchain';
import { FailoverService } from './services/failover';

interface AppConfig {
    port: number;
    wsPort: number;
    mongoUrl: string;
    redisUrl: string;
    influxUrl: string;
    influxToken: string;
    influxOrg: string;
    influxBucket: string;
    jwtSecret: string;
    nodeEnv: string;
    logLevel: string;
}

class A2ZAPIServer {
    private app: express.Application;
    private server: Server;
    private wsServer: WebSocketServer;
    private config: AppConfig;
    private db: Db;
    private redis: Redis;
    private influx: InfluxDB;
    private logger: winston.Logger;
    private metrics: prometheus.Registry;

    // Services
    private networkService: NetworkService;
    private frerService: FRERService;
    private monitoringService: MonitoringService;
    private mlService: MLService;
    private blockchainService: BlockchainService;
    private failoverService: FailoverService;

    constructor() {
        this.config = this.loadConfig();
        this.app = express();
        this.setupLogger();
        this.setupMetrics();
    }

    private loadConfig(): AppConfig {
        return {
            port: parseInt(process.env.API_PORT || '3000'),
            wsPort: parseInt(process.env.WS_PORT || '8080'),
            mongoUrl: process.env.MONGO_URL || 'mongodb://localhost:27017/a2z',
            redisUrl: process.env.REDIS_URL || 'redis://localhost:6379',
            influxUrl: process.env.INFLUX_URL || 'http://localhost:8086',
            influxToken: process.env.INFLUX_TOKEN || 'a2z-influx-token',
            influxOrg: process.env.INFLUX_ORG || 'a2z',
            influxBucket: process.env.INFLUX_BUCKET || 'tsn_metrics',
            jwtSecret: process.env.JWT_SECRET || 'a2z-jwt-secret-2024',
            nodeEnv: process.env.NODE_ENV || 'development',
            logLevel: process.env.LOG_LEVEL || 'info'
        };
    }

    private setupLogger(): void {
        this.logger = winston.createLogger({
            level: this.config.logLevel,
            format: winston.format.combine(
                winston.format.timestamp(),
                winston.format.errors({ stack: true }),
                winston.format.json()
            ),
            transports: [
                new winston.transports.Console({
                    format: winston.format.combine(
                        winston.format.colorize(),
                        winston.format.simple()
                    )
                }),
                new winston.transports.File({
                    filename: 'logs/error.log',
                    level: 'error'
                }),
                new winston.transports.File({
                    filename: 'logs/combined.log'
                })
            ]
        });
    }

    private setupMetrics(): void {
        this.metrics = new prometheus.Registry();
        prometheus.collectDefaultMetrics({ register: this.metrics });

        // Custom metrics
        const httpDuration = new prometheus.Histogram({
            name: 'http_request_duration_seconds',
            help: 'Duration of HTTP requests in seconds',
            labelNames: ['method', 'route', 'status'],
            buckets: [0.1, 0.5, 1, 2, 5]
        });

        const wsConnections = new prometheus.Gauge({
            name: 'websocket_connections',
            help: 'Number of active WebSocket connections'
        });

        const frerRecoveries = new prometheus.Counter({
            name: 'frer_recoveries_total',
            help: 'Total number of FRER recoveries',
            labelNames: ['stream_id', 'severity']
        });

        const anomaliesDetected = new prometheus.Counter({
            name: 'anomalies_detected_total',
            help: 'Total number of anomalies detected',
            labelNames: ['model', 'severity']
        });

        this.metrics.registerMetric(httpDuration);
        this.metrics.registerMetric(wsConnections);
        this.metrics.registerMetric(frerRecoveries);
        this.metrics.registerMetric(anomaliesDetected);
    }

    private async connectDatabases(): Promise<void> {
        try {
            // MongoDB connection
            const mongoClient = new MongoClient(this.config.mongoUrl);
            await mongoClient.connect();
            this.db = mongoClient.db();
            this.logger.info('Connected to MongoDB');

            // Redis connection
            this.redis = new Redis(this.config.redisUrl);
            this.redis.on('connect', () => {
                this.logger.info('Connected to Redis');
            });

            // InfluxDB connection
            this.influx = new InfluxDB({
                url: this.config.influxUrl,
                token: this.config.influxToken
            });
            this.logger.info('Connected to InfluxDB');

        } catch (error) {
            this.logger.error('Database connection failed:', error);
            throw error;
        }
    }

    private initializeServices(): void {
        this.networkService = new NetworkService(this.db, this.redis, this.logger);
        this.frerService = new FRERService(this.db, this.redis, this.influx, this.logger);
        this.monitoringService = new MonitoringService(this.influx, this.redis, this.logger);
        this.mlService = new MLService(this.db, this.redis, this.logger);
        this.blockchainService = new BlockchainService(this.db, this.logger);
        this.failoverService = new FailoverService(this.db, this.redis, this.logger);

        this.logger.info('All services initialized');
    }

    private setupMiddleware(): void {
        // Security middleware
        this.app.use(helmet({
            contentSecurityPolicy: {
                directives: {
                    defaultSrc: ["'self'"],
                    styleSrc: ["'self'", "'unsafe-inline'"],
                    scriptSrc: ["'self'", "'unsafe-inline'"],
                    imgSrc: ["'self'", 'data:', 'https:']
                }
            }
        }));

        // CORS configuration
        this.app.use(cors({
            origin: process.env.CORS_ORIGIN?.split(',') || '*',
            credentials: true,
            methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
            allowedHeaders: ['Content-Type', 'Authorization', 'X-API-Key']
        }));

        // Compression
        this.app.use(compression());

        // Body parsing
        this.app.use(express.json({ limit: '10mb' }));
        this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));

        // Rate limiting
        const limiter = rateLimit({
            windowMs: 15 * 60 * 1000, // 15 minutes
            max: 100, // limit each IP to 100 requests per windowMs
            message: 'Too many requests from this IP',
            standardHeaders: true,
            legacyHeaders: false
        });
        this.app.use('/api/', limiter);

        // Request ID and logging
        this.app.use((req: Request, res: Response, next: NextFunction) => {
            req.id = uuidv4();
            next();
        });
        this.app.use(requestLogger(this.logger));

        // Metrics middleware
        this.app.use(metricsMiddleware(this.metrics));

        this.logger.info('Middleware configured');
    }

    private setupRoutes(): void {
        // Health check
        this.app.get('/health', (req: Request, res: Response) => {
            res.json({
                status: 'healthy',
                timestamp: new Date().toISOString(),
                uptime: process.uptime(),
                memory: process.memoryUsage(),
                version: process.env.npm_package_version || '2.0.0'
            });
        });

        // Readiness check
        this.app.get('/ready', async (req: Request, res: Response) => {
            try {
                // Check database connections
                await this.db.admin().ping();
                await this.redis.ping();
                
                res.json({
                    status: 'ready',
                    services: {
                        mongodb: 'connected',
                        redis: 'connected',
                        influxdb: 'connected'
                    }
                });
            } catch (error) {
                res.status(503).json({
                    status: 'not_ready',
                    error: error.message
                });
            }
        });

        // Metrics endpoint
        this.app.get('/metrics', async (req: Request, res: Response) => {
            res.set('Content-Type', this.metrics.contentType);
            const metrics = await this.metrics.metrics();
            res.end(metrics);
        });

        // API documentation
        const swaggerDocument = YAML.load(path.join(__dirname, 'openapi-spec.yaml'));
        this.app.use('/api-docs', swaggerUi.serve, swaggerUi.setup(swaggerDocument, {
            customCss: '.swagger-ui .topbar { display: none }',
            customSiteTitle: 'A2Z TSN/FRER API Documentation'
        }));

        // API v2 routes
        const apiRouter = express.Router();

        // Apply authentication middleware to protected routes
        apiRouter.use(authMiddleware(this.config.jwtSecret));

        // Mount route handlers
        apiRouter.use('/network', NetworkRouter(this.networkService));
        apiRouter.use('/frer', FRERRouter(this.frerService));
        apiRouter.use('/monitoring', MonitoringRouter(this.monitoringService));
        apiRouter.use('/ml', MLRouter(this.mlService));
        apiRouter.use('/blockchain', BlockchainRouter(this.blockchainService));
        apiRouter.use('/configuration', ConfigurationRouter(this.networkService));
        apiRouter.use('/failover', FailoverRouter(this.failoverService));

        this.app.use('/v2', apiRouter);

        // Error handling
        this.app.use(errorHandler(this.logger));

        // 404 handler
        this.app.use((req: Request, res: Response) => {
            res.status(404).json({
                error: 'Not Found',
                message: `Route ${req.method} ${req.path} not found`,
                timestamp: new Date().toISOString()
            });
        });

        this.logger.info('Routes configured');
    }

    private setupWebSocket(): void {
        this.wsServer = new WebSocketServer({
            port: this.config.wsPort,
            perMessageDeflate: {
                zlibDeflateOptions: {
                    chunkSize: 1024,
                    memLevel: 7,
                    level: 3
                },
                zlibInflateOptions: {
                    chunkSize: 10 * 1024
                },
                clientNoContextTakeover: true,
                serverNoContextTakeover: true,
                serverMaxWindowBits: 10,
                concurrencyLimit: 10,
                threshold: 1024
            }
        });

        this.wsServer.on('connection', (ws, req) => {
            const clientId = uuidv4();
            this.logger.info(`WebSocket client connected: ${clientId}`);

            // Send initial connection message
            ws.send(JSON.stringify({
                type: 'connection',
                clientId,
                timestamp: new Date().toISOString()
            }));

            // Handle incoming messages
            ws.on('message', async (data) => {
                try {
                    const message = JSON.parse(data.toString());
                    await this.handleWebSocketMessage(ws, message);
                } catch (error) {
                    this.logger.error('WebSocket message error:', error);
                    ws.send(JSON.stringify({
                        type: 'error',
                        error: 'Invalid message format'
                    }));
                }
            });

            // Handle disconnection
            ws.on('close', () => {
                this.logger.info(`WebSocket client disconnected: ${clientId}`);
            });

            // Handle errors
            ws.on('error', (error) => {
                this.logger.error(`WebSocket error for client ${clientId}:`, error);
            });

            // Setup heartbeat
            const heartbeat = setInterval(() => {
                if (ws.readyState === ws.OPEN) {
                    ws.send(JSON.stringify({
                        type: 'heartbeat',
                        timestamp: new Date().toISOString()
                    }));
                } else {
                    clearInterval(heartbeat);
                }
            }, 30000);
        });

        this.logger.info(`WebSocket server listening on port ${this.config.wsPort}`);
    }

    private async handleWebSocketMessage(ws: any, message: any): Promise<void> {
        switch (message.type) {
            case 'subscribe':
                // Subscribe to specific data streams
                if (message.stream === 'metrics') {
                    this.subscribeToMetrics(ws);
                } else if (message.stream === 'alerts') {
                    this.subscribeToAlerts(ws);
                } else if (message.stream === 'frer') {
                    this.subscribeToFRER(ws);
                }
                break;

            case 'command':
                // Handle real-time commands
                const result = await this.executeCommand(message.command, message.params);
                ws.send(JSON.stringify({
                    type: 'command_result',
                    commandId: message.commandId,
                    result
                }));
                break;

            case 'query':
                // Handle real-time queries
                const data = await this.executeQuery(message.query, message.params);
                ws.send(JSON.stringify({
                    type: 'query_result',
                    queryId: message.queryId,
                    data
                }));
                break;

            default:
                ws.send(JSON.stringify({
                    type: 'error',
                    error: 'Unknown message type'
                }));
        }
    }

    private subscribeToMetrics(ws: any): void {
        const interval = setInterval(async () => {
            if (ws.readyState !== ws.OPEN) {
                clearInterval(interval);
                return;
            }

            const metrics = await this.monitoringService.getCurrentMetrics();
            ws.send(JSON.stringify({
                type: 'metrics',
                data: metrics,
                timestamp: new Date().toISOString()
            }));
        }, 5000);
    }

    private subscribeToAlerts(ws: any): void {
        // Subscribe to Redis pub/sub for alerts
        const subscriber = this.redis.duplicate();
        subscriber.subscribe('alerts');

        subscriber.on('message', (channel, message) => {
            if (ws.readyState === ws.OPEN) {
                ws.send(JSON.stringify({
                    type: 'alert',
                    data: JSON.parse(message),
                    timestamp: new Date().toISOString()
                }));
            } else {
                subscriber.unsubscribe();
                subscriber.quit();
            }
        });
    }

    private subscribeToFRER(ws: any): void {
        // Subscribe to FRER events
        const subscriber = this.redis.duplicate();
        subscriber.subscribe('frer_events');

        subscriber.on('message', (channel, message) => {
            if (ws.readyState === ws.OPEN) {
                ws.send(JSON.stringify({
                    type: 'frer_event',
                    data: JSON.parse(message),
                    timestamp: new Date().toISOString()
                }));
            } else {
                subscriber.unsubscribe();
                subscriber.quit();
            }
        });
    }

    private async executeCommand(command: string, params: any): Promise<any> {
        switch (command) {
            case 'trigger_frer_recovery':
                return await this.frerService.triggerRecovery(params.streamId, params.reason);
            case 'execute_failover':
                return await this.failoverService.executeFailover(params.componentId, params.targetNode);
            case 'acknowledge_alert':
                return await this.monitoringService.acknowledgeAlert(params.alertId);
            default:
                throw new Error(`Unknown command: ${command}`);
        }
    }

    private async executeQuery(query: string, params: any): Promise<any> {
        switch (query) {
            case 'switch_status':
                return await this.networkService.getSwitchStatus(params.switchId);
            case 'stream_statistics':
                return await this.frerService.getStreamStatistics(params.streamId);
            case 'anomaly_details':
                return await this.mlService.getAnomalyDetails(params.anomalyId);
            default:
                throw new Error(`Unknown query: ${query}`);
        }
    }

    public async start(): Promise<void> {
        try {
            // Connect to databases
            await this.connectDatabases();

            // Initialize services
            this.initializeServices();

            // Setup Express app
            this.setupMiddleware();
            this.setupRoutes();

            // Start HTTP server
            this.server = this.app.listen(this.config.port, () => {
                this.logger.info(`API server listening on port ${this.config.port}`);
                this.logger.info(`Environment: ${this.config.nodeEnv}`);
                this.logger.info(`API documentation available at http://localhost:${this.config.port}/api-docs`);
            });

            // Setup WebSocket server
            this.setupWebSocket();

            // Graceful shutdown handling
            process.on('SIGTERM', () => this.shutdown());
            process.on('SIGINT', () => this.shutdown());

        } catch (error) {
            this.logger.error('Failed to start server:', error);
            process.exit(1);
        }
    }

    private async shutdown(): Promise<void> {
        this.logger.info('Shutting down gracefully...');

        // Close WebSocket server
        this.wsServer.close();

        // Close HTTP server
        this.server.close();

        // Close database connections
        await this.redis.quit();
        
        this.logger.info('Server shutdown complete');
        process.exit(0);
    }
}

// Extend Express Request type
declare global {
    namespace Express {
        interface Request {
            id?: string;
            user?: any;
        }
    }
}

// Start the server
if (require.main === module) {
    const server = new A2ZAPIServer();
    server.start().catch((error) => {
        console.error('Failed to start server:', error);
        process.exit(1);
    });
}

export default A2ZAPIServer;