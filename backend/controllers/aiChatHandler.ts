import { Request, Response } from 'express';
import { Ollama } from 'ollama';
import fs from 'fs';
import path from 'path';
import csv from 'csv-parser';
import natural from 'natural';
import { fileURLToPath } from 'url';
import { dirname } from 'path';
import { LRUCache } from 'lru-cache';
import pLimit from 'p-limit';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const TfIdf = natural.TfIdf;

// ML Pipeline Paths
const ML_DIR = path.join(__dirname, '../ml');
const PROCESSED_DATA_PATH = path.join(ML_DIR, 'processed_data.csv');
const TRAINING_METRICS_PATH = path.join(ML_DIR, 'training_metrics.json');

// Initialize Ollama client with optimized settings
const ollama = new Ollama({
  host: process.env.OLLAMA_HOST || 'http://localhost:11434',
});

// Chat model configuration - using gemma:2b for speed
const CHAT_MODEL = 'gemma:2b';
let isChatModelWarmedUp = false;

// Warm up the chat model on startup
const warmUpChatModel = async () => {
  if (isChatModelWarmedUp) return;
  
  try {
    console.log(`[Chat] Warming up ${CHAT_MODEL} model...`);
    const startTime = Date.now();
    
    await ollama.chat({
      model: CHAT_MODEL,
      messages: [{ role: 'user', content: 'Hi' }],
      options: {
        num_predict: 5,
        temperature: 0.1,
        num_ctx: 512,
      },
    });
    
    const duration = Date.now() - startTime;
    console.log(`[Chat] Model warmed up in ${duration}ms`);
    isChatModelWarmedUp = true;
  } catch (error: any) {
    console.warn(`[Chat] Model warm-up failed:`, error.message);
  }
};

// Initialize chat warm-up
warmUpChatModel();

// Food data storage (from ML pipeline)
let foods: any[] = [];
let tfidf: any = null;
let mlMetrics: any = null;
let mlFeatures: string[] = [];

// OPTIMIZATION 3: Enhanced LRU Cache for faster chat responses
const responseCache = new LRUCache<string, string>({
  max: 200,                    // Increased cache size
  ttl: 1000 * 60 * 20,        // 20 minutes TTL
});

// Additional cache for common queries
const quickResponseCache = new LRUCache<string, string>({
  max: 50,                     // Cache for frequent questions
  ttl: 1000 * 60 * 60,        // 1 hour for common responses
});

// OPTIMIZATION 2: Concurrency limiter using p-limit (prevents OOM/swapping)
const limit = pLimit(1); // Max 1 concurrent request to prevent memory issues

// Load ML processed dataset with fitness scores
export async function loadUSDAData() {
  console.log('[Chat] Loading ML processed data...');
  
  // Load processed foods from ML pipeline
  if (!fs.existsSync(PROCESSED_DATA_PATH)) {
    console.warn('[Chat] ML processed data not found. Run: python backend/ml/preprocess.py');
    foods = [];
    return;
  }

  return new Promise<void>((resolve, reject) => {
    const rows: any[] = [];
    fs.createReadStream(PROCESSED_DATA_PATH)
      .pipe(csv())
      .on('data', (row) => {
        // Load foods with ML features and fitness labels
        const food = {
          food_name: row.food_name,
          category: row.category,
          // Raw nutrients
          calories: parseFloat(row.calories) || 0,
          protein: parseFloat(row.protein) || 0,
          carbs: parseFloat(row.carbs) || 0,
          fat: parseFloat(row.fat) || 0,
          iron: parseFloat(row.iron) || 0,
          vitamin_c: parseFloat(row.vitamin_c) || 0,
          // ML engineered features
          nutrient_density: parseFloat(row.nutrient_density) || 0,
          protein_ratio: parseFloat(row.protein_ratio) || 0,
          carb_fat_ratio: parseFloat(row.carb_fat_ratio) || 0,
          energy_density: parseFloat(row.energy_density) || 0,
          micronutrient_score: parseFloat(row.micronutrient_score) || 0,
          // Binary flags
          is_glutenfree: parseInt(row.is_glutenfree) || 0,
          is_nutfree: parseInt(row.is_nutfree) || 0,
          is_vegan: parseInt(row.is_vegan) || 0,
          // ML fitness label (0=unfit, 1=fit)
          fit: parseInt(row.fit) || 0,
        };
        if (food.food_name) {
          rows.push(food);
        }
      })
      .on('end', () => {
        foods = rows;
        console.log(`[Chat] Loaded ${foods.length} foods from ML pipeline`);
        console.log(`[Chat] ML features available: nutrient_density, protein_ratio, micronutrient_score, etc.`);
        
        // Load ML training metrics
        try {
          const metricsData = fs.readFileSync(TRAINING_METRICS_PATH, 'utf-8');
          mlMetrics = JSON.parse(metricsData);
          console.log(`[Chat] Loaded ML metrics: F1=${(mlMetrics.metrics.test.f1_macro * 100).toFixed(1)}%, Accuracy=${(mlMetrics.metrics.test.accuracy * 100).toFixed(1)}%`);
        } catch (err) {
          console.warn('[Chat] Could not load ML metrics:', err);
        }

        buildTfIdf();
        resolve();
      })
      .on('error', reject);
  });
}

function buildTfIdf() {
  tfidf = new TfIdf();
  foods.forEach((food) => {
    // Include food name, category, and ML fitness score in searchable text
    const fitnessLabel = food.fit === 1 ? 'healthy fit nutritious' : '';
    const text = `${food.food_name || ''} ${food.category || ''} ${fitnessLabel}`.toLowerCase();
    tfidf.addDocument(text);
  });
  console.log('[Chat] Built TF-IDF index with ML fitness scores');
}

function searchFoods(query: string, limit: number = 5): any[] {
  if (!tfidf || foods.length === 0) return [];

  const scores: Array<{ index: number; score: number }> = [];
  tfidf.tfidfs(query.toLowerCase(), (i: number, score: number) => {
    if (score > 0) {
      scores.push({ index: i, score });
    }
  });

  // PRIORITIZE ML-FIT FOODS: Filter for fit=1 first, fallback to all if insufficient
  const fitScores = scores.filter(s => foods[s.index].fit === 1);
  const selectedScores = fitScores.length >= limit ? fitScores : scores;
  
  // Sort by TF-IDF score, then by ML fitness score
  selectedScores.sort((a, b) => {
    const scoreDiff = b.score - a.score;
    if (Math.abs(scoreDiff) < 0.01) {
      // If TF-IDF scores are similar, prefer ML-fit foods
      return (foods[b.index].fit || 0) - (foods[a.index].fit || 0);
    }
    return scoreDiff;
  });
  
  const results = selectedScores.slice(0, limit).map((s) => ({
    ...foods[s.index],
    tfidf_score: s.score.toFixed(3),
  }));
  
  console.log(`[Chat] RAG Search: Found ${fitScores.length} ML-fit foods, returning ${results.filter(r => r.fit === 1).length} fit out of ${results.length} total`);
  
  return results;
}

// Helper function to create cache key
function createCacheKey(message: string, context: string): string {
  return `${message.toLowerCase()}-${context.slice(0, 100)}`;
}

export const chat = async (req: Request, res: Response) => {
  console.time('chat-response'); // OPTIMIZATION: Detailed timing
  const t0 = Date.now();
  
  try {
    const message = req.body?.message ? String(req.body.message) : '';
    const stream = req.body?.stream === true;

    if (!message.trim()) {
      return res.status(400).json({
        success: false,
        error: 'Message is required',
      });
    }

    console.log('[Chat] 📝 Query received:', message.substring(0, 50) + '...');
    console.time('rag-search');

    // OPTIMIZATION 4: RAG with Top-5 foods (ML-enhanced results)
    // If no good matches, search for general healthy foods
    let ragRows = searchFoods(message, 5);
    
    // Fallback: If no ML-fit foods found, search for "fruit vegetable protein" 
    if (ragRows.filter(r => r.fit === 1).length === 0) {
      console.log('[Chat] No fit foods found, using general healthy food search');
      ragRows = searchFoods('fruit vegetable protein healthy', 5);
    }
    
    console.timeEnd('rag-search');
    
    // Build ML-enhanced context (internal use - foods already ranked by ML fitness)
    const mlContext = ragRows.map((food, idx) => {
      return `${idx + 1}. ${food.food_name} (${food.category})\n` +
             `   Calories: ${food.calories}kcal | Protein: ${food.protein}g | Vitamin C: ${food.vitamin_c}mg\n` +
             `   ${food.is_vegan ? 'Vegan-friendly' : ''} ${food.is_glutenfree ? 'Gluten-free' : ''} ${food.is_nutfree ? 'Nut-free' : ''}`;
    }).join('\n\n');
    
    const context = mlContext;
    console.log('[Chat] ML Context for Ollama (foods pre-ranked by ML):\n', mlContext.substring(0, 300) + '...');

    // OPTIMIZATION 3: Enhanced cache checking (quick cache first, then regular)
    const cacheKey = createCacheKey(message, context);
    const quickCacheKey = message.toLowerCase().trim();
    
    // Check quick response cache first (for common questions)
    let cached = quickResponseCache.get(quickCacheKey);
    if (cached) {
      console.log('[Chat] ⚡ Quick cache hit!');
      console.timeEnd('chat-response');
      return res.json({
        success: true,
        response: cached,
        cached: true,
        cacheType: 'quick',
        ms: Date.now() - t0,
      });
    }

    // Check regular response cache
    cached = responseCache.get(cacheKey);
    if (cached) {
      console.log('[Chat] ⚡ Regular cache hit!');
      console.timeEnd('chat-response');
      return res.json({
        success: true,
        response: cached,
        cached: true,
        cacheType: 'regular',
        ms: Date.now() - t0,
      });
    }

    // System prompt - conversational and user-friendly
    const system = `You are a friendly nutrition assistant. You have access to a carefully curated database of nutritious foods. When recommending foods, explain their health benefits in simple terms - focus on what nutrients they provide, how they can help with specific goals (like weight loss, muscle building, heart health), and why they're good choices. Be conversational and encouraging. Avoid technical jargon.`;
    
    // User prompt - request natural, conversational response
    const userPrompt = context.length > 0 
      ? `User Question: ${message}\n\n[Top Recommended Foods for This Goal]\n${context}\n\nBased on the foods above (already ranked by nutritional quality), provide a friendly, conversational response that:\n- Recommends 3-5 foods from the list above\n- Explains their health benefits in simple terms (what nutrients they have and why that matters)\n- Suggests how to incorporate them into meals\n- Uses an encouraging, supportive tone\n\nDo NOT mention: ML scores, fitness scores, models, technical metrics. Just talk naturally about the foods and their benefits.`
      : message;
    
    console.log('[Chat] User prompt preview:', userPrompt.substring(0, 400) + '...');

    // OPTIMIZATION 6: Streaming responses (SSE)
    if (stream) {
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      });

      console.log('[Chat]  Sending query to Ollama (streaming)...');
      console.time('ollama-streaming');

      let fullResponse = '';

      try {
        // Ensure chat model is warmed up
        await warmUpChatModel();
        
        // OPTIMIZATION 2: Concurrency limiting with p-limit
        await limit(async () => {
          try {
            const response = await ollama.chat({
            model: CHAT_MODEL,
            messages: [
              { role: 'system', content: system },
              { role: 'user', content: userPrompt },
            ],
            stream: true,
            options: {
              num_predict: 500,    // Increased for complete responses
              temperature: 0.7,
              num_ctx: 1024,
              top_p: 0.9,
              top_k: 20,
            },
          });

          let chunkCount = 0;
          for await (const chunk of response) {
            const content = chunk.message?.content || '';
            if (content) {
              fullResponse += content;
              chunkCount++;
              res.write(`data: ${JSON.stringify({ content })}\n\n`);
            }
          }
          
          console.log(`[Chat] Complete response: ${fullResponse.length} chars, ${chunkCount} chunks`);
          
          // Retry if response seems incomplete (less than 200 chars)
          if (fullResponse.length < 200) {
            console.log('[Chat] Response seems incomplete, retrying with higher num_predict...');
            fullResponse = '';
            chunkCount = 0;
            
            const retryResponse = await ollama.chat({
              model: CHAT_MODEL,
              messages: [
                { role: 'system', content: system },
                { role: 'user', content: userPrompt },
              ],
              stream: true,
              options: {
                num_predict: 600,    // Higher for complete retry
                temperature: 0.5,
                num_ctx: 1024,
                top_p: 0.85,
                top_k: 20,
                repeat_penalty: 1.1,
              },
            });
            
            for await (const chunk of retryResponse) {
              const content = chunk.message?.content || '';
              if (content) {
                fullResponse += content;
                chunkCount++;
                res.write(`data: ${JSON.stringify({ content })}\n\n`);
              }
            }
            
            console.log(`[Chat] Retry complete: ${fullResponse.length} chars, ${chunkCount} chunks`);
          }
          } catch (ollamaError: any) {
            console.error('[Chat] Ollama streaming error:', ollamaError);
            throw new Error(`AI service unavailable: ${ollamaError.message}`);
          }
        });

        res.write('data: [DONE]\n\n');
        res.end();

        console.timeEnd('ollama-streaming');
        console.timeEnd('chat-response');
        console.log(`[Chat] ✅ Stream complete. Total time: ${Date.now() - t0}ms, Final response: ${fullResponse.length} chars`);

        // Cache the complete response in both caches
        responseCache.set(cacheKey, fullResponse);
        
        // Also cache in quick cache if it's a short, common question
        if (message.length < 50 && fullResponse.length < 200) {
          quickResponseCache.set(quickCacheKey, fullResponse);
        }
        
        console.log(`[Chat] Response includes ${ragRows.length} ML-analyzed foods`);
      } catch (streamError: any) {
        console.error('[Chat] Streaming error:', streamError);
        res.write(`data: ${JSON.stringify({ error: streamError.message })}\n\n`);
        res.end();
      }
    } else {
      // Ensure chat model is warmed up
      await warmUpChatModel();
      
      // Non-streaming response (optimized)
      console.log('[Chat] 🤖 Sending query to Ollama (non-streaming, optimized)...');
      console.time('ollama-request');

      // OPTIMIZATION 2: Concurrency limiting with p-limit
      const response = await limit(() =>
        ollama.chat({
          model: CHAT_MODEL,
          messages: [
            { role: 'system', content: system },
            { role: 'user', content: userPrompt },
          ],
          options: {
            num_predict: 500,    // Increased for complete responses
            temperature: 0.7,
            num_ctx: 1024,
            top_p: 0.9,
            top_k: 20,
          },
        })
      ) as any;

      console.timeEnd('ollama-request');

      const content =
        response?.message?.content ||
        'Sorry, I could not generate a response right now.';
      
      console.log(`[Chat] ✅ Non-streaming complete: ${content.length} chars`);
      console.log('[Chat] ✅ Ollama response received:', content.substring(0, 100) + '...');
      console.timeEnd('chat-response');
      console.log(`[Chat] ⏱️  Total response time: ${Date.now() - t0}ms`);

      // Cache the response in both caches
      responseCache.set(cacheKey, content);
      
      // Also cache in quick cache if it's a short, common question
      if (message.length < 50 && content.length < 200) {
        quickResponseCache.set(quickCacheKey, content);
      }

      res.json({
        success: true,
        response: content,
        cached: false,
        ms: Date.now() - t0,
        mlFoods: ragRows.length, // Number of ML-analyzed foods used
      });
    }
  } catch (error: any) {
    console.error('[Chat] Error:', error);
    console.timeEnd('chat-response');
    res.status(500).json({
      success: false,
      error: error.message || 'Internal server error',
    });
  }
};

// Export cache management functions
export const getCacheStats = () => {
  return {
    responseCache: {
      size: responseCache.size,
      maxSize: responseCache.max,
    },
    quickResponseCache: {
      size: quickResponseCache.size,
      maxSize: quickResponseCache.max,
    },
  };
};

export const clearCaches = () => {
  responseCache.clear();
  quickResponseCache.clear();
  console.log('[Chat] All caches cleared (regular + quick)');
};
