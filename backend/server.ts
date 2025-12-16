import express, { Request, Response, NextFunction } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import mongoose from 'mongoose';
import dotenv from 'dotenv';
import { fileURLToPath } from 'url';
import { dirname, join } from 'path';

// ESM dirname equivalent
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Load environment variables
dotenv.config({ path: join(__dirname, '../.env') });

// Import routes
import authRoutes from './routes/auth';
import apiRoutes from './routes/api';
import recommendationRoutes from './routes/recommendations';
import mealPlanRoutes from './routes/mealPlan';

// Import chat controller to initialize data
import { loadUSDAData } from './controllers/aiChatHandler';

const app = express();
const PORT = process.env.PORT || 5000;
const MONGO_URI = process.env.MONGO_URI || 'mongodb://localhost:27017/nutrisolve';

// Startup banner
console.log('\n' + '='.repeat(60));
console.log('NutriSolve Backend Server Starting...');
console.log('='.repeat(60) + '\n');

// Middleware
console.log('[Setup] Configuring middleware...');
app.use(helmet({
  crossOriginResourcePolicy: { policy: "cross-origin" },
  crossOriginOpenerPolicy: { policy: "same-origin-allow-popups" },
}));
console.log('   Helmet security headers enabled');

app.use(cors({
  origin: function (origin, callback) {
    // Allow requests with no origin (like mobile apps, curl, postman)
    if (!origin) return callback(null, true);
    
    // In development, allow all localhost and 127.0.0.1 origins
    if (process.env.NODE_ENV !== 'production') {
      if (origin.startsWith('http://localhost:') || 
          origin.startsWith('http://127.0.0.1:') ||
          origin.startsWith('https://localhost:') ||
          origin.startsWith('https://127.0.0.1:')) {
        return callback(null, true);
      }
    }
    
    // In production, check against FRONTEND_URL
    const allowedOrigins = [
      process.env.FRONTEND_URL,
      'http://localhost:5173',
      'http://localhost:5174',
      'http://localhost:8080',
      'http://localhost:8082',
      'http://127.0.0.1:5173',
      'http://127.0.0.1:5174'
    ].filter(Boolean);
    
    if (allowedOrigins.includes(origin)) {
      callback(null, true);
    } else {
      console.warn(`[CORS] Blocked origin: ${origin}`);
      callback(null, true); // Allow in development, log warning
    }
  },
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization'],
}));
console.log('   CORS configured for development');

app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));
console.log('   Body parsers configured (10MB limit)');

// Set server timeout for long-running AI operations (10 minutes)
app.use((req, res, next) => {
  req.setTimeout(600000); // 10 minutes
  res.setTimeout(600000); // 10 minutes
  next();
});
console.log('   Request timeout set to 10 minutes for AI operations');

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // Limit each IP to 100 requests per windowMs
  message: 'Too many requests from this IP, please try again later.',
});
app.use('/api', limiter);
console.log('   Rate limiting: 100 requests per 15 minutes\n');

// Connect to MongoDB
console.log('[Database] Connecting to MongoDB...');
console.log(`   URI: ${MONGO_URI.replace(/\/\/.*@/, '//<credentials>@')}`);

mongoose
  .connect(MONGO_URI)
  .then(() => {
    console.log('   MongoDB connected successfully');
    console.log(`   Database: ${mongoose.connection.name}`);
    console.log(`   Host: ${mongoose.connection.host}:${mongoose.connection.port}\n`);
  })
  .catch((err) => {
    console.error('   MongoDB connection failed:', err.message);
    console.error('   Please ensure MongoDB is running and accessible\n');
    process.exit(1);
  });

// Load USDA data
console.log('[Data] Loading USDA nutrition database...');
loadUSDAData()
  .then(() => {
    console.log('   USDA data loaded successfully\n');
  })
  .catch((err) => {
    console.warn('   Failed to load USDA data:', err.message);
    console.warn('   AI chat may have limited nutrition data\n');
  });

// Handle preflight requests for all routes
app.use((req, res, next) => {
  if (req.method === 'OPTIONS') {
    res.header('Access-Control-Allow-Origin', req.headers.origin || '*');
    res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
    res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    res.header('Access-Control-Allow-Credentials', 'true');
    return res.sendStatus(200);
  }
  next();
});

// Routes
console.log('[Routes] Registering API endpoints...');
app.use('/api/auth', authRoutes);
console.log('   /api/auth - Authentication routes');

app.use('/api', apiRoutes);
console.log('   /api - General API routes');

app.use('/api', recommendationRoutes);
console.log('   /api - Recommendation routes');

app.use('/api/meal-plan', mealPlanRoutes);
console.log('   /api/meal-plan - Meal planning routes\n');

// Health check
app.get('/health', (req: Request, res: Response) => {
  res.json({
    success: true,
    message: 'NutriSolve API is running',
    timestamp: new Date().toISOString(),
  });
});

// Error handling middleware
app.use((err: any, req: Request, res: Response, next: NextFunction) => {
  console.error('[Server] Error:', err);
  res.status(err.status || 500).json({
    success: false,
    error: err.message || 'Internal server error',
  });
});

// Start server
app.listen(PORT, () => {
  console.log('='.repeat(60));
  console.log('Server Started Successfully!');
  console.log('='.repeat(60));
  console.log(`\nServer URL: http://localhost:${PORT}`);
  console.log(`Environment: ${process.env.NODE_ENV || 'development'}`);
  console.log(`Started at: ${new Date().toLocaleString()}`);
  console.log(`\nAvailable Endpoints:`);
  console.log(`   • Health Check:  http://localhost:${PORT}/health`);
  console.log(`   • Auth API:      http://localhost:${PORT}/api/auth`);
  console.log(`   • Main API:      http://localhost:${PORT}/api`);
  console.log(`   • Meal Plans:    http://localhost:${PORT}/api/meal-plan`);
  console.log(`\nPress Ctrl+C to stop the server`);
  console.log('='.repeat(60) + '\n');
});

export default app;
