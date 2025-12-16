/**
 * Recommendation Controller
 * Integrates Python ML prediction service with TypeScript backend
 * 
 * Endpoints:
 * - POST /api/recommend - Get personalized meal recommendations
 * - POST /api/recommend-with-chat - Get recommendations + Ollama response
 * 
 * Integration Flow:
 * 1. Receive user profile + query from frontend
 * 2. Validate input data
 * 3. Spawn Python predict.py process via child_process
 * 4. Pass JSON input via stdin
 * 5. Parse JSON output from stdout
 * 6. (Optional) Feed top recommendations to Ollama for natural language response
 * 7. Return combined response to frontend
 */

import { Request, Response } from 'express';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

// ESM dirname equivalent
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Path to ML prediction script
const ML_DIR = path.join(__dirname, '..', 'ml');
const PREDICT_SCRIPT = path.join(ML_DIR, 'predict.py');

/**
 * Interface for user profile from onboarding
 */
interface UserProfile {
  age: number;
  gender: string;
  weight: number;
  height: number;
  activityLevel: string;
  primaryGoal: string;
  dietaryRestrictions: string[];
  weeklyBudget: number;
  favoriteCuisines: string[];
  mealFrequency: string;
}

/**
 * Interface for recommendation request
 */
interface RecommendationRequest {
  userProfile: UserProfile;
  query: string;
  top_k?: number;
}

/**
 * Interface for ML prediction response
 */
interface MLRecommendation {
  name: string;
  category: string;
  fit_score: number;
  confidence: string;
  nutrition: {
    calories: number;
    protein: number;
    carbs: number;
    fat: number;
    fiber: number;
    sugars: number;
  };
  cost: number;
  reasons: string[];
  dietary_info: string[];
}

interface MLResponse {
  recommendations: MLRecommendation[];
  query: string;
  total_eligible: number;
  model_version: string;
  user_goal: string;
  message?: string;
  error?: string;
}

/**
 * Execute Python ML prediction script
 * 
 * @param input - User profile and query
 * @returns Promise with ML predictions
 */
async function runMLPrediction(input: RecommendationRequest): Promise<MLResponse> {
  return new Promise((resolve, reject) => {
    // Spawn Python process
    const python = spawn('python3', [PREDICT_SCRIPT], {
      cwd: ML_DIR,
    });

    let stdout = '';
    let stderr = '';

    // Collect stdout (JSON response)
    python.stdout.on('data', (data) => {
      stdout += data.toString();
    });

    // Collect stderr (errors)
    python.stderr.on('data', (data) => {
      stderr += data.toString();
    });

    // Handle process completion
    python.on('close', (code) => {
      if (code !== 0) {
        console.error('[RecController] Python script error:', stderr);
        
        // Try to parse error JSON from stderr
        try {
          const errorJson = JSON.parse(stderr);
          reject(new Error(errorJson.error || 'ML prediction failed'));
        } catch {
          reject(new Error(`ML prediction failed with code ${code}: ${stderr}`));
        }
        return;
      }

      // Parse JSON response
      try {
        const result: MLResponse = JSON.parse(stdout);
        resolve(result);
      } catch (error) {
        console.error('[RecController] Failed to parse ML output:', stdout);
        reject(new Error('Failed to parse ML prediction response'));
      }
    });

    // Handle process errors
    python.on('error', (error) => {
      console.error('[RecController] Failed to spawn Python process:', error);
      reject(new Error(`Failed to run ML prediction: ${error.message}`));
    });

    // Send input data to Python via stdin
    try {
      python.stdin.write(JSON.stringify(input));
      python.stdin.end();
    } catch (error) {
      reject(new Error('Failed to send data to ML prediction script'));
    }
  });
}

/**
 * POST /api/recommend
 * Get personalized meal recommendations using ML model
 */
export const getRecommendations = async (req: Request, res: Response) => {
  try {
    console.log('[RecController] Received recommendation request');

    // Extract data from request
    const { userProfile, query, top_k } = req.body;

    // Validate required fields
    if (!userProfile) {
      return res.status(400).json({
        error: 'Missing userProfile. Please complete onboarding first.',
      });
    }

    if (!query || typeof query !== 'string') {
      return res.status(400).json({
        error: 'Missing or invalid query string',
      });
    }

    // Prepare input for ML model
    const mlInput: RecommendationRequest = {
      userProfile,
      query,
      top_k: top_k || 5,
    };

    console.log('[RecController] Calling ML prediction service...');
    console.log('[RecController] User goal:', userProfile.primaryGoal);
    console.log('[RecController] Dietary restrictions:', userProfile.dietaryRestrictions);

    // Run ML prediction
    const mlResult = await runMLPrediction(mlInput);

    console.log('[RecController] ML prediction successful');
    console.log('[RecController] Returned', mlResult.recommendations?.length || 0, 'recommendations');

    // Return recommendations
    return res.status(200).json({
      success: true,
      data: mlResult,
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error('[RecController] Error in getRecommendations:', error);

    // Check if ML models need training
    if (error.message?.includes('not found') || error.message?.includes('Run train.py')) {
      return res.status(503).json({
        error: 'ML models not ready. Please run: python backend/ml/preprocess.py && python backend/ml/train.py',
        details: error.message,
      });
    }

    return res.status(500).json({
      error: 'Failed to generate recommendations',
      details: error.message,
    });
  }
};

/**
 * POST /api/recommend-with-chat
 * Get recommendations + generate natural language response via Ollama
 * 
 * This endpoint:
 * 1. Gets ML rankings (same as /api/recommend)
 * 2. Formats rankings as context for Ollama
 * 3. Sends combined prompt to Ollama chat
 * 4. Returns both structured recommendations and natural response
 */
export const getRecommendationsWithChat = async (req: Request, res: Response) => {
  try {
    console.log('[RecController] Received recommendation + chat request');

    const { userProfile, query, top_k } = req.body;

    // Validate
    if (!userProfile || !query) {
      return res.status(400).json({
        error: 'Missing userProfile or query',
      });
    }

    // Prepare ML input
    const mlInput: RecommendationRequest = {
      userProfile,
      query,
      top_k: top_k || 5,
    };

    // Run ML prediction
    console.log('[RecController] Getting ML recommendations...');
    const mlResult = await runMLPrediction(mlInput);

    // Format recommendations as context for Ollama
    const contextLines = [
      `[ML Recommendation Rankings for: ${query}]`,
      `User Goal: ${userProfile.primaryGoal}`,
      `Dietary Restrictions: ${userProfile.dietaryRestrictions.join(', ') || 'None'}`,
      `Budget: $${userProfile.weeklyBudget}/week`,
      '',
      'Top Recommended Meals:',
    ];

    mlResult.recommendations.forEach((rec, idx) => {
      contextLines.push(
        `${idx + 1}. ${rec.name} (Fit Score: ${(rec.fit_score * 100).toFixed(0)}%, Confidence: ${rec.confidence})`
      );
      contextLines.push(`   Category: ${rec.category}`);
      contextLines.push(`   Nutrition: ${rec.nutrition.calories} kcal, ${rec.nutrition.protein}g protein, ${rec.nutrition.fiber}g fiber`);
      contextLines.push(`   Cost: $${rec.cost.toFixed(2)} per serving`);
      if (rec.reasons.length > 0) {
        contextLines.push(`   Why: ${rec.reasons.join(', ')}`);
      }
      if (rec.dietary_info.length > 0) {
        contextLines.push(`   Tags: ${rec.dietary_info.join(', ')}`);
      }
      contextLines.push('');
    });

    contextLines.push('[End of Rankings]');
    contextLines.push('');
    contextLines.push('Please provide a friendly, personalized response explaining these recommendations and how they support the user\'s goal.');

    const contextPrompt = contextLines.join('\n');

    // Integrate with Ollama for natural language response
    console.log('[RecController] Sending ML results to Ollama for natural language generation...');
    
    let ollamaResponse = null;
    try {
      // Import Ollama dynamically
      const { Ollama } = await import('ollama');
      const ollama = new Ollama({
        host: process.env.OLLAMA_HOST || 'http://localhost:11434',
      });
      
      const response = await ollama.chat({
        model: 'gemma:2b',
        messages: [
          { 
            role: 'system', 
            content: 'You are a nutrition coach. Explain meal recommendations in a friendly, motivating way. Be concise (2-3 paragraphs).' 
          },
          { 
            role: 'user', 
            content: contextPrompt 
          },
        ],
        options: {
          num_predict: 200,
          temperature: 0.7,
          num_ctx: 1024,
        },
      });
      
      ollamaResponse = response?.message?.content || null;
      console.log('[RecController] Ollama response generated successfully');
    } catch (ollamaError: any) {
      console.warn('[RecController] Ollama unavailable, returning ML results only:', ollamaError.message);
    }

    // Return combined response with ML rankings + Ollama explanation
    return res.status(200).json({
      success: true,
      data: {
        recommendations: mlResult,
        mlContext: contextPrompt,
        chatResponse: ollamaResponse,
      },
      timestamp: new Date().toISOString(),
    });
  } catch (error: any) {
    console.error('[RecController] Error in getRecommendationsWithChat:', error);

    return res.status(500).json({
      error: 'Failed to generate recommendations with chat',
      details: error.message,
    });
  }
};

/**
 * GET /api/recommend/status
 * Check if ML models are ready
 */
export const checkMLStatus = async (req: Request, res: Response) => {
  try {
    const fs = await import('fs');
    const modelPath = path.join(ML_DIR, 'rf_model.pkl');
    const preprocessorPath = path.join(ML_DIR, 'preprocessor.pkl');
    const metricsPath = path.join(ML_DIR, 'training_metrics.json');

    const modelExists = fs.existsSync(modelPath);
    const preprocessorExists = fs.existsSync(preprocessorPath);
    const metricsExists = fs.existsSync(metricsPath);

    const ready = modelExists && preprocessorExists;

    let metrics = null;
    if (metricsExists) {
      const metricsData = fs.readFileSync(metricsPath, 'utf-8');
      metrics = JSON.parse(metricsData);
    }

    return res.status(200).json({
      ready,
      components: {
        model: modelExists,
        preprocessor: preprocessorExists,
        metrics: metricsExists,
      },
      metrics: metrics ? {
        test_f1: metrics.metrics?.test?.f1_macro,
        test_accuracy: metrics.metrics?.test?.accuracy,
        model: metrics.model,
        training_date: metrics.training_date,
      } : null,
      instructions: ready
        ? 'ML models are ready!'
        : 'Run: python backend/ml/preprocess.py && python backend/ml/train.py',
    });
  } catch (error: any) {
    console.error('[RecController] Error checking ML status:', error);
    return res.status(500).json({
      error: 'Failed to check ML status',
      details: error.message,
    });
  }
};

/**
 * POST /api/recommend/train
 * Trigger ML model training (use with caution)
 * Requires admin authentication in production
 */
export const trainMLModel = async (req: Request, res: Response) => {
  try {
    console.log('[RecController] Training ML model...');

    // Run preprocessing
    const preprocessSpawn = spawn('python3', [path.join(ML_DIR, 'preprocess.py')], {
      cwd: ML_DIR,
    });

    let preprocessOutput = '';
    preprocessSpawn.stdout.on('data', (data) => {
      preprocessOutput += data.toString();
      console.log('[Preprocess]', data.toString());
    });

    preprocessSpawn.stderr.on('data', (data) => {
      console.error('[Preprocess Error]', data.toString());
    });

    await new Promise((resolve, reject) => {
      preprocessSpawn.on('close', (code) => {
        if (code !== 0) {
          reject(new Error('Preprocessing failed'));
        } else {
          resolve(true);
        }
      });
    });

    // Run training
    const trainSpawn = spawn('python3', [path.join(ML_DIR, 'train.py')], {
      cwd: ML_DIR,
    });

    let trainOutput = '';
    trainSpawn.stdout.on('data', (data) => {
      trainOutput += data.toString();
      console.log('[Train]', data.toString());
    });

    trainSpawn.stderr.on('data', (data) => {
      console.error('[Train Error]', data.toString());
    });

    await new Promise((resolve, reject) => {
      trainSpawn.on('close', (code) => {
        if (code !== 0) {
          reject(new Error('Training failed'));
        } else {
          resolve(true);
        }
      });
    });

    console.log('[RecController] Training complete!');

    return res.status(200).json({
      success: true,
      message: 'ML model training completed successfully',
      output: {
        preprocess: preprocessOutput,
        train: trainOutput,
      },
    });
  } catch (error: any) {
    console.error('[RecController] Training error:', error);
    return res.status(500).json({
      error: 'ML model training failed',
      details: error.message,
    });
  }
};
