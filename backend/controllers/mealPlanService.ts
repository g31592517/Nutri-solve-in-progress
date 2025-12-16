import { Request, Response } from 'express';
import { Ollama } from 'ollama';
import Tesseract from 'tesseract.js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { dirname } from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// ML Pipeline Paths
const ML_DIR = path.join(__dirname, '../ml');
const PROCESSED_DATA_PATH = path.join(ML_DIR, 'processed_data.csv');
const TRAINING_METRICS_PATH = path.join(ML_DIR, 'training_metrics.json');

// ML data storage
let mlFoods: any[] = [];
let mlMetrics: any = null;
let mlFitFoods: any[] = []; // Only ML-recommended (fit=1) foods

// Initialize Ollama client with optimized settings
const ollama = new Ollama({
  host: process.env.OLLAMA_HOST || 'http://localhost:11434',
});

// Use faster model for meal planning (force gemma:2b for speed)
const getMealPlanModel = () => {
  return 'gemma:2b'; // Force fastest model for quick inference
};

// Model warm-up and caching
let isModelWarmedUp = false;
let cachedResponses = new Map();
const CACHE_DURATION = 30 * 60 * 1000; // 30 minutes

// Warm up the model on startup
const warmUpModel = async () => {
  if (isModelWarmedUp) return;
  
  try {
    const model = getMealPlanModel();
    console.log(`[MealPlan] Warming up ${model} model...`);
    const startTime = Date.now();
    
    await ollama.chat({
      model,
      messages: [{ role: 'user', content: 'Ready' }],
      options: {
        num_predict: 10,
        temperature: 0.1,
        num_ctx: 512, // Minimal context for warm-up
      },
    });
    
    const duration = Date.now() - startTime;
    console.log(`[MealPlan] Model warmed up in ${duration}ms`);
    isModelWarmedUp = true;
  } catch (error: any) {
    console.warn('[MealPlan] Model warm-up failed:', error.message);
  }
};

// Get ML-recommended foods by category and dietary restrictions
const getMLFitFoods = (category: string, restrictions: string[] = []) => {
  let foods = mlFitFoods.filter(f => 
    f.category && f.category.toLowerCase().includes(category.toLowerCase())
  );
  
  // Apply dietary filters
  if (restrictions.includes('vegan') || restrictions.includes('Vegan')) {
    foods = foods.filter(f => f.is_vegan === 1);
  }
  if (restrictions.includes('gluten-free') || restrictions.includes('Gluten Free')) {
    foods = foods.filter(f => f.is_glutenfree === 1);
  }
  if (restrictions.includes('nut-free') || restrictions.includes('Nut Allergy')) {
    foods = foods.filter(f => f.is_nutfree === 1);
  }
  
  return foods;
};

// Generate cache key for meal plan requests
const generateCacheKey = (profile: any, budget: string, preferences: string, varietyMode: string) => {
  return JSON.stringify({ profile, budget, preferences, varietyMode });
};

// Get cached response if available and not expired
const getCachedResponse = (cacheKey: string) => {
  const cached = cachedResponses.get(cacheKey);
  if (cached && Date.now() - cached.timestamp < CACHE_DURATION) {
    console.log('[MealPlan] Using cached response');
    return cached.data;
  }
  return null;
};

// Cache response
const setCachedResponse = (cacheKey: string, data: any) => {
  cachedResponses.set(cacheKey, {
    data,
    timestamp: Date.now()
  });
  
  // Clean old cache entries
  if (cachedResponses.size > 100) {
    const oldestKey = cachedResponses.keys().next().value;
    cachedResponses.delete(oldestKey);
  }
};

// Load ML processed foods for meal planning
const loadMLFoods = async () => {
  if (!fs.existsSync(PROCESSED_DATA_PATH)) {
    console.warn('[MealPlan] ML data not found. Run: python backend/ml/preprocess.py');
    return;
  }
  
  const csv = await import('csv-parser');
  return new Promise<void>((resolve) => {
    const rows: any[] = [];
    fs.createReadStream(PROCESSED_DATA_PATH)
      .pipe(csv.default())
      .on('data', (row: any) => {
        const food = {
          food_name: row.food_name,
          category: row.category,
          calories: parseFloat(row.calories) || 0,
          protein: parseFloat(row.protein) || 0,
          carbs: parseFloat(row.carbs) || 0,
          fat: parseFloat(row.fat) || 0,
          iron: parseFloat(row.iron) || 0,
          vitamin_c: parseFloat(row.vitamin_c) || 0,
          nutrient_density: parseFloat(row.nutrient_density) || 0,
          micronutrient_score: parseFloat(row.micronutrient_score) || 0,
          is_glutenfree: parseInt(row.is_glutenfree) || 0,
          is_nutfree: parseInt(row.is_nutfree) || 0,
          is_vegan: parseInt(row.is_vegan) || 0,
          fit: parseInt(row.fit) || 0, // ML fitness label
        };
        rows.push(food);
      })
      .on('end', () => {
        mlFoods = rows;
        mlFitFoods = rows.filter(f => f.fit === 1); // Only ML-recommended foods
        console.log(`[MealPlan] Loaded ${mlFoods.length} foods from ML pipeline (${mlFitFoods.length} fit)`);
        
        // Load metrics
        try {
          const metricsData = fs.readFileSync(TRAINING_METRICS_PATH, 'utf-8');
          mlMetrics = JSON.parse(metricsData);
          console.log(`[MealPlan] ML Model: F1=${(mlMetrics.metrics.test.f1_macro * 100).toFixed(1)}%`);
        } catch (err) {
          console.warn('[MealPlan] Could not load ML metrics');
        }
        
        resolve();
      });
  });
};

// Initialize warm-up and ML data loading
warmUpModel();
loadMLFoods();

// Generate weekly meal plan via Ollama with streaming support
export const generateMealPlanStream = async (req: Request, res: Response) => {
  try {
    const { profile, budget, preferences, varietyMode } = req.body;

    if (!profile) {
      return res.status(400).json({
        success: false,
        error: 'User profile is required',
      });
    }

    // Set up streaming headers
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      'Connection': 'keep-alive',
    });

    // Build comprehensive system prompt
    const systemPrompt = `You are a professional nutritionist and meal planning expert. Generate detailed, realistic meal plans based on user profiles and constraints. Generate the meal plan day by day, providing complete JSON for each day as you go. Always respond with valid JSON only, no additional text.`;

    // Build user prompt with all context
    const goalMapping: any = {
      'weight_loss': 'Weight Loss (calorie deficit, high protein)',
      'muscle_gain': 'Muscle Gain (high protein, calorie surplus)',
      'maintenance': 'Maintain Weight (balanced nutrition)',
      'general_health': 'General Health (balanced, nutrient-dense)'
    };

    const activityMapping: any = {
      'sedentary': 'Sedentary (little to no exercise)',
      'light': 'Light Activity (1-3 days/week)',
      'moderate': 'Moderate Activity (3-5 days/week)',
      'active': 'Active (6-7 days/week)',
      'very_active': 'Very Active (intense daily exercise)'
    };

    const budgetRanges: any = {
      '20-50': '$20-50 per week',
      '50-100': '$50-100 per week',
      '100-150': '$100-150 per week',
      '150+': '$150+ per week'
    };

    const goal = goalMapping[profile.primaryGoal] || 'General Health';
    const activity = activityMapping[profile.activityLevel] || 'Moderate';
    const budgetText = budgetRanges[budget] || '$50-100 per week';
    const restrictions = profile.dietaryRestrictions || [];
    const varietyText = varietyMode === 'consistent' 
      ? 'Keep meals consistent across days (meal prep friendly, some repeats allowed)'
      : 'Maximize variety with different meals each day';

    const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
    let completedDays: any[] = [];
    
    // Send initial status
    res.write(`data: ${JSON.stringify({ 
      type: 'status', 
      message: 'Starting meal plan generation...', 
      progress: 0 
    })}\n\n`);

    // Ensure model is warmed up
    await warmUpModel();

    // Generate each day progressively
    for (let i = 0; i < days.length; i++) {
      const day = days[i];
      
      // Send progress update
      res.write(`data: ${JSON.stringify({ 
        type: 'status', 
        message: `Generating ${day}...`, 
        progress: Math.round((i / days.length) * 100) 
      })}\n\n`);

      const dayPrompt = `Create ${day} meal plan for: ${goal}, ${restrictions.length > 0 ? restrictions.join('/') : 'no restrictions'}, ${budgetText}, ${preferences || 'balanced meals'}.

Return ONLY valid JSON:
{
  "day": "${day}",
  "meals": [
    {
      "type": "breakfast",
      "name": "Specific meal name",
      "calories": 300,
      "protein": 15,
      "carbs": 35,
      "fat": 10,
      "ingredients": ["ingredient1", "ingredient2", "ingredient3"]
    },
    {
      "type": "lunch", 
      "name": "Specific meal name",
      "calories": 400,
      "protein": 25,
      "carbs": 40,
      "fat": 15,
      "ingredients": ["ingredient1", "ingredient2", "ingredient3"]
    },
    {
      "type": "dinner",
      "name": "Specific meal name", 
      "calories": 350,
      "protein": 20,
      "carbs": 30,
      "fat": 12,
      "ingredients": ["ingredient1", "ingredient2", "ingredient3"]
    }
  ]
}`;

      try {
        console.log(`[MealPlan] Calling Gemma for ${day}...`);
        const startTime = Date.now();
        
        // Generate each meal individually for true progressive rendering
        const mealTypes = ['breakfast', 'lunch', 'dinner'];
        const dayMeals: any[] = [];
        
        for (const mealType of mealTypes) {
          console.log(`[MealPlan] 🤖 Calling REAL Gemma AI for ${day} ${mealType}...`);
          
          // Get ML-recommended foods for this meal type with variety
          // Rotate through different foods to avoid repetition
          const startIndex = (i * mealTypes.indexOf(mealType) + mealTypes.indexOf(mealType) * 3) % mlFitFoods.length;
          const mlSuggestions = mlFitFoods.slice(startIndex, startIndex + 10).map(f => 
            `${f.food_name} (${f.category}, ${f.calories}kcal, ${f.protein}g protein)`
          ).join(', ');
          
          // Get category-specific foods for better variety
          const categoryMap: any = {
            'breakfast': ['fruit', 'grain', 'dairy'],
            'lunch': ['protein', 'vegetable', 'grain'],
            'dinner': ['protein', 'vegetable']
          };
          const relevantCategories = categoryMap[mealType] || [];
          const categoryFoods = mlFitFoods.filter(f => 
            relevantCategories.some((cat: string) => f.category && f.category.toLowerCase().includes(cat))
          ).slice(0, 8);
          
          const categorySuggestions = categoryFoods.length > 0 
            ? `\n[Category-Specific Options]: ${categoryFoods.map(f => f.food_name).join(', ')}`
            : '';
          
          // Simple prompt focusing on ML-recommended ingredients
          const mealPrompt = `Create ${mealType} for ${goal}.

Base Ingredients (ML-recommended): ${mlSuggestions}

Use these healthy ingredients. ${restrictions.length > 0 ? restrictions.join('/') + ' only.' : ''} Return JSON only:
{"type":"${mealType}","name":"meal name","calories":300,"protein":15,"carbs":35,"fat":10,"ingredients":["item1","item2","item3"]}`;

          let meal;
          try {
            const gemmaStartTime = Date.now();
            // NO TIMEOUT - Let Gemma complete naturally
            const response = await ollama.chat({
              model: getMealPlanModel(),
              messages: [
                { role: 'system', content: 'Nutritionist. JSON only.' },
                { role: 'user', content: mealPrompt },
              ],
              options: { 
                num_predict: 100,
                temperature: 0.5,
                num_ctx: 512,
                top_k: 30,
                top_p: 0.9,
              },
            });
            
            const gemmaTime = Date.now() - gemmaStartTime;
            const content = (response as any)?.message?.content || '';
            console.log(`[MealPlan] ✅ Gemma responded in ${gemmaTime}ms`);
            console.log(`[MealPlan] 📝 Raw Gemma output: ${content.substring(0, 200)}...`);
            
            // Parse meal from Gemma response
            try {
              const jsonMatch = content.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
              const jsonString = jsonMatch ? jsonMatch[1] : content;
              meal = JSON.parse(jsonString.trim());
              console.log(`[MealPlan] ✅ Successfully parsed Gemma JSON for ${day} ${mealType}`);
            } catch (parseError) {
              console.log(`[MealPlan] ⚠️ Parse error, using simple default`);
              // Use simple default from ML foods
              const mlFood = mlFitFoods[Math.floor(Math.random() * mlFitFoods.length)];
              meal = {
                type: mealType,
                name: `${mlFood.food_name} Bowl`,
                calories: mlFood.calories,
                protein: mlFood.protein,
                carbs: mlFood.carbs || 30,
                fat: mlFood.fat || 10,
                ingredients: [mlFood.food_name, "mixed vegetables", "olive oil"]
              };
            }
            
          } catch (gemmaError: any) {
            console.log(`[MealPlan] ❌ Gemma error: ${gemmaError.message}`);
            // Use simple ML-based default
            const mlFood = mlFitFoods[Math.floor(Math.random() * mlFitFoods.length)];
            meal = {
              type: mealType,
              name: `${mlFood.food_name} Bowl`,
              calories: mlFood.calories,
              protein: mlFood.protein,
              carbs: mlFood.carbs || 30,
              fat: mlFood.fat || 10,
              ingredients: [mlFood.food_name, "mixed vegetables", "olive oil"]
            };
          }
          
          dayMeals.push(meal);
          
          // Send meal immediately
          res.write(`data: ${JSON.stringify({ 
            type: 'meal', 
            meal: meal
          })}\n\n`);
          
          console.log(`[MealPlan] 📤 Sent ${day} ${mealType}: ${meal.name}`);
        }
        
        const duration = Date.now() - startTime;
        console.log(`[MealPlan] Completed ${day} in ${duration}ms`);

        // Calculate day totals
        const dayTotals = dayMeals.reduce(
          (acc: any, meal: any) => ({
            totalCalories: acc.totalCalories + (meal.calories || 0),
            totalProtein: acc.totalProtein + (meal.protein || 0),
            totalCarbs: acc.totalCarbs + (meal.carbs || 0),
            totalFat: acc.totalFat + (meal.fat || 0),
          }),
          { totalCalories: 0, totalProtein: 0, totalCarbs: 0, totalFat: 0 }
        );

        const completeDayPlan = {
          day: day,
          meals: dayMeals,
          ...dayTotals,
        };

        completedDays.push(completeDayPlan);

        // Send day complete signal
        res.write(`data: ${JSON.stringify({ 
          type: 'day_complete',
          progress: Math.round(((i + 1) / days.length) * 100)
        })}\n\n`);

      } catch (error: any) {
        console.error(`[MealPlan] Error generating ${day}:`, error);
        
        // Skip this day and continue
        res.write(`data: ${JSON.stringify({ 
          type: 'error',
          message: `Failed to generate ${day}, skipping...`
        })}\n\n`);
      }
    }

    // Calculate weekly totals
    const weeklyTotals = completedDays.reduce(
      (acc: any, day: any) => ({
        calories: acc.calories + day.totalCalories,
        protein: acc.protein + day.totalProtein,
        carbs: acc.carbs + day.totalCarbs,
        fat: acc.fat + day.totalFat,
        estimatedCost: acc.estimatedCost + day.meals.reduce((sum: number, m: any) => sum + (m.cost || 0), 0),
      }),
      { calories: 0, protein: 0, carbs: 0, fat: 0, estimatedCost: 0 }
    );

    const finalMealPlan = {
      days: completedDays,
      weeklyTotals,
      metadata: {
        createdAt: new Date().toISOString(),
        generationType: 'ai',
        userGoal: profile.primaryGoal,
        budget,
        preferences,
      },
    };

    // Send final complete meal plan
    res.write(`data: ${JSON.stringify({ 
      type: 'complete', 
      mealPlan: finalMealPlan,
      progress: 100
    })}\n\n`);

    res.write('data: [DONE]\n\n');
    res.end();

  } catch (error: any) {
    console.error('[MealPlan] Streaming error:', error);
    res.write(`data: ${JSON.stringify({ 
      type: 'error', 
      message: error.message || 'Failed to generate meal plan' 
    })}\n\n`);
    res.end();
  }
};

// Generate weekly meal plan via Ollama
export const generateMealPlan = async (req: Request, res: Response) => {
  try {
    const { profile, budget, preferences, varietyMode } = req.body;

    if (!profile) {
      return res.status(400).json({
        success: false,
        error: 'User profile is required',
      });
    }

    // Build comprehensive system prompt
    const systemPrompt = `You are a professional nutritionist and meal planning expert. Generate detailed, realistic meal plans based on user profiles and constraints. Always respond with valid JSON only, no additional text.`;

    // Build user prompt with all context
    const goalMapping: any = {
      'weight_loss': 'Weight Loss (calorie deficit, high protein)',
      'muscle_gain': 'Muscle Gain (high protein, calorie surplus)',
      'maintenance': 'Maintain Weight (balanced nutrition)',
      'general_health': 'General Health (balanced, nutrient-dense)'
    };

    const activityMapping: any = {
      'sedentary': 'Sedentary (little to no exercise)',
      'light': 'Light Activity (1-3 days/week)',
      'moderate': 'Moderate Activity (3-5 days/week)',
      'active': 'Active (6-7 days/week)',
      'very_active': 'Very Active (intense daily exercise)'
    };

    const budgetRanges: any = {
      '20-50': '$20-50 per week',
      '50-100': '$50-100 per week',
      '100-150': '$100-150 per week',
      '150+': '$150+ per week'
    };

    const goal = goalMapping[profile.primaryGoal] || 'General Health';
    const activity = activityMapping[profile.activityLevel] || 'Moderate';
    const budgetText = budgetRanges[budget] || '$50-100 per week';
    const restrictions = profile.dietaryRestrictions || [];
    const varietyText = varietyMode === 'consistent' 
      ? 'Keep meals consistent across days (meal prep friendly, some repeats allowed)'
      : 'Maximize variety with different meals each day';

    // Get ML-recommended foods for context
    const mlTopFoods = mlFitFoods.slice(0, 15).map(f => 
      `${f.food_name} (${f.calories}kcal, ${f.protein}g protein, ML score: ${f.nutrient_density.toFixed(2)})`
    ).join('\n- ');
    
    const mlContext = mlFitFoods.length > 0 
      ? `\n\n**ML-Recommended Ingredients (use these when possible):**\n- ${mlTopFoods}\n\nOur ML model analyzed 205 foods and rated these as optimal for nutrition.`
      : '';
    
    const userPrompt = `Generate a complete 7-day meal plan for the following profile:

**User Profile:**
- Age: ${profile.age || 'Not specified'}
- Gender: ${profile.gender || 'Not specified'}
- Weight: ${profile.weight || 'Not specified'}kg
- Activity Level: ${activity}
- Primary Goal: ${goal}
- Dietary Restrictions: ${restrictions.length > 0 ? restrictions.join(', ') : 'None'}

**Meal Plan Requirements:**
- Weekly Budget: ${budgetText}
- Special Preferences: ${preferences || 'None'}
- Variety Mode: ${varietyText}${mlContext}

**Output Format (strict JSON):**
{
  "days": [
    {
      "day": "Monday",
      "meals": [
        {
          "type": "breakfast",
          "name": "Meal name",
          "calories": 350,
          "protein": 20,
          "carbs": 40,
          "fat": 12,
          "ingredients": ["ingredient1", "ingredient2", "..."],
          "cost": 3.50,
          "prepTime": 15
        },
        {
          "type": "lunch",
          "name": "...",
          "calories": 500,
          "protein": 35,
          "carbs": 45,
          "fat": 18,
          "ingredients": ["..."],
          "cost": 5.00,
          "prepTime": 20
        },
        {
          "type": "dinner",
          "name": "...",
          "calories": 600,
          "protein": 40,
          "carbs": 50,
          "fat": 20,
          "ingredients": ["..."],
          "cost": 7.00,
          "prepTime": 30
        },
        {
          "type": "snack",
          "name": "...",
          "calories": 150,
          "protein": 10,
          "carbs": 15,
          "fat": 5,
          "ingredients": ["..."],
          "cost": 2.00,
          "prepTime": 5
        }
      ]
    }
  ]
}

Repeat this structure for all 7 days (Monday through Sunday). Ensure:
1. Meals align with the user's goal (e.g., weight loss = lower calories, muscle gain = high protein)
2. Honor all dietary restrictions strictly
3. Stay within budget per meal
4. Include realistic ingredients and prep times
5. Provide accurate macro estimates

Generate the meal plan now in JSON format only:`;

    // Check cache first
    const cacheKey = generateCacheKey(profile, budget, preferences, varietyMode);
    const cachedPlan = getCachedResponse(cacheKey);
    
    if (cachedPlan) {
      return res.json({
        success: true,
        mealPlan: cachedPlan,
        cached: true
      });
    }

    // Ensure model is warmed up
    await warmUpModel();
    
    // Call Ollama with optimized settings and error handling
    console.log('[MealPlan] Generating meal plan via Ollama (optimized)...');
    
    let response;
    try {
      // NO TIMEOUT - Let Ollama complete full week meal plan
      response = await ollama.chat({
        model: getMealPlanModel(),
        messages: [
          { role: 'system', content: systemPrompt },
          { role: 'user', content: userPrompt },
        ],
        options: { 
          num_predict: 3000,
          temperature: 0.4,
          num_ctx: 2048,
          top_k: 25,
          top_p: 0.85,
          repeat_penalty: 1.1,
          seed: -1,
        },
      });
    } catch (ollamaError: any) {
      console.error('[MealPlan] Ollama connection error:', ollamaError);
      return res.status(500).json({
        success: false,
        error: `AI service unavailable: ${ollamaError.message}. Please ensure Ollama is running.`,
      });
    }

    const content = response?.message?.content || '';
    console.log('[MealPlan] Ollama response received, parsing JSON...');

    // Extract JSON from response (handle markdown code blocks)
    let planJson;
    try {
      // Try to extract JSON from markdown code blocks
      const jsonMatch = content.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
      const jsonString = jsonMatch ? jsonMatch[1] : content;
      planJson = JSON.parse(jsonString.trim());
    } catch (parseError) {
      console.error('[MealPlan] Failed to parse JSON:', parseError);
      return res.status(500).json({
        success: false,
        error: 'Failed to parse meal plan from AI response',
        rawResponse: content,
      });
    }

    // Calculate totals for each day and weekly aggregate
    const daysWithTotals = planJson.days.map((day: any) => {
      const dayTotals = day.meals.reduce(
        (acc: any, meal: any) => ({
          totalCalories: acc.totalCalories + (meal.calories || 0),
          totalProtein: acc.totalProtein + (meal.protein || 0),
          totalCarbs: acc.totalCarbs + (meal.carbs || 0),
          totalFat: acc.totalFat + (meal.fat || 0),
        }),
        { totalCalories: 0, totalProtein: 0, totalCarbs: 0, totalFat: 0 }
      );

      return {
        ...day,
        ...dayTotals,
      };
    });

    const weeklyTotals = daysWithTotals.reduce(
      (acc: any, day: any) => ({
        calories: acc.calories + day.totalCalories,
        protein: acc.protein + day.totalProtein,
        carbs: acc.carbs + day.totalCarbs,
        fat: acc.fat + day.totalFat,
        estimatedCost: acc.estimatedCost + day.meals.reduce((sum: number, m: any) => sum + (m.cost || 0), 0),
      }),
      { calories: 0, protein: 0, carbs: 0, fat: 0, estimatedCost: 0 }
    );

    const mealPlan = {
      days: daysWithTotals,
      weeklyTotals,
      metadata: {
        createdAt: new Date().toISOString(),
        generationType: 'ai',
        userGoal: profile.primaryGoal,
        budget,
        preferences,
      },
    };

    // Cache the successful response
    setCachedResponse(cacheKey, mealPlan);

    res.json({
      success: true,
      mealPlan,
      cached: false
    });
  } catch (error: any) {
    console.error('[MealPlan] Error:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'Failed to generate meal plan',
    });
  }
};

// Swap a single meal via Ollama
export const swapMeal = async (req: Request, res: Response) => {
  try {
    const { mealName, mealType, day, profile, budget, preferences } = req.body;

    const systemPrompt = `You are a friendly nutrition assistant. Suggest tasty meal alternatives using similar food categories. Be creative with combinations. Respond with valid JSON only.`;

    // Get ML-fit alternatives from SAME category as original meal
    // This ensures variety while maintaining nutritional profile
    const originalCategory = mealName.toLowerCase().includes('salad') ? 'vegetable' 
      : mealName.toLowerCase().includes('chicken') || mealName.toLowerCase().includes('beef') ? 'protein'
      : mealName.toLowerCase().includes('pasta') || mealName.toLowerCase().includes('rice') ? 'grain'
      : 'mixed';
    
    const similarFoods = mlFitFoods.filter(f => {
      const category = f.category ? f.category.toLowerCase() : '';
      return originalCategory === 'mixed' || category.includes(originalCategory) || category.includes('vegetable');
    }).slice(0, 12);
    
    const mlAlternatives = similarFoods.map(f => 
      `${f.food_name} (${f.category}, ${f.calories}kcal, ${f.protein}g protein)`
    ).join(', ');
    
    const userPrompt = `Create 3 delicious alternative meals for "${mealName}" (${mealType}).

User Profile:
- Goal: ${profile.primaryGoal || 'general health'}
- Dietary Restrictions: ${profile.dietaryRestrictions?.join(', ') || 'None'}
- Budget: ${budget || '$50-100/week'}
- Preferences: ${preferences || 'None'}

[Similar Nutritious Ingredients]: ${mlAlternatives}

IMPORTANT: Use ingredients from the list above to create SIMILAR but DIFFERENT meals. Keep the same meal type (${mealType}) and similar calorie/protein range, but make each alternative unique and tasty.

Output format (JSON only):
{
  "alternatives": [
    {
      "name": "Alternative meal name",
      "calories": 300,
      "protein": 20,
      "carbs": 30,
      "fat": 10,
      "ingredients": ["ing1", "ing2"],
      "reason": "Why this is a good swap"
    }
  ]
}`;

    // Ensure model is warmed up
    await warmUpModel();
    
    // NO TIMEOUT - Let Ollama suggest similar foods to completion
    const response = await ollama.chat({
      model: getMealPlanModel(),
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt },
      ],
      options: { 
        num_predict: 400,
        temperature: 0.5,
        num_ctx: 1024,
        top_k: 20,
        top_p: 0.85,
      },
    });

    const content = response?.message?.content || '';
    const jsonMatch = content.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
    const jsonString = jsonMatch ? jsonMatch[1] : content;
    const alternatives = JSON.parse(jsonString.trim());

    res.json({
      success: true,
      alternatives: alternatives.alternatives || [],
    });
  } catch (error: any) {
    console.error('[MealPlan] Swap error:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'Failed to generate meal swaps',
    });
  }
};

// Extract preferences from uploaded text/doc
export const extractPreferences = async (req: Request, res: Response) => {
  try {
    const { text } = req.body;

    if (!text) {
      return res.status(400).json({
        success: false,
        error: 'Text content is required',
      });
    }

    const systemPrompt = `You are a food preference analyzer. Extract dietary preferences, restrictions, and requests from text. Respond with JSON only.`;

    const userPrompt = `Analyze this text and extract food preferences:

"${text}"

Output format (JSON only):
{
  "preferences": ["likes/preferences as short phrases"],
  "avoids": ["foods to avoid"],
  "requests": ["specific meal requests or patterns"]
}

Example:
{
  "preferences": ["pasta", "Mediterranean cuisine", "quick meals"],
  "avoids": ["dairy", "shellfish"],
  "requests": ["light breakfasts", "high-protein dinners"]
}`;

    // Ensure model is warmed up
    await warmUpModel();
    
    // NO TIMEOUT - Let preference extraction complete
    const response = await ollama.chat({
      model: getMealPlanModel(),
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt },
      ],
      options: { 
        num_predict: 200,
        temperature: 0.3,
        num_ctx: 1024,
        top_k: 20,
        top_p: 0.8,
      },
    });

    const content = response?.message?.content || '';
    const jsonMatch = content.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
    const jsonString = jsonMatch ? jsonMatch[1] : content;
    const extracted = JSON.parse(jsonString.trim());

    res.json({
      success: true,
      extracted,
    });
  } catch (error: any) {
    console.error('[MealPlan] Extract error:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'Failed to extract preferences',
    });
  }
};

// OCR from uploaded image
export const ocrImage = async (req: Request, res: Response) => {
  try {
    if (!req.file) {
      return res.status(400).json({
        success: false,
        error: 'Image file is required',
      });
    }

    const imagePath = req.file.path;
    console.log('[MealPlan] Running OCR on image:', imagePath);

    const result = await Tesseract.recognize(imagePath, 'eng', {
      logger: (m) => console.log('[OCR]', m),
    });

    const text = result.data.text;

    // Clean up uploaded file
    fs.unlinkSync(imagePath);

    // Now extract preferences from OCR text
    const systemPrompt = `Extract food preferences from OCR text (may have errors). Respond with JSON only.`;
    const userPrompt = `This text was extracted from an image (OCR). Extract food preferences:

"${text}"

Output JSON format:
{
  "preferences": ["food likes"],
  "avoids": ["foods to avoid"],
  "requests": ["meal patterns"]
}`;

    // Ensure model is warmed up
    await warmUpModel();
    
    const response = await ollama.chat({
      model: getMealPlanModel(),
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt },
      ],
      options: { 
        num_predict: 250,
        temperature: 0.3,
        num_ctx: 1024,
        top_k: 15,
        top_p: 0.8,
      },
    });

    const content = response?.message?.content || '';
    const jsonMatch = content.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
    const jsonString = jsonMatch ? jsonMatch[1] : content;
    const extracted = JSON.parse(jsonString.trim());

    res.json({
      success: true,
      ocrText: text,
      extracted,
    });
  } catch (error: any) {
    console.error('[MealPlan] OCR error:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'Failed to process image',
    });
  }
};

// Generate AI insights for current plan
export const generateInsights = async (req: Request, res: Response) => {
  try {
    const { plan, profile } = req.body;

    if (!plan || !plan.days) {
      return res.status(400).json({
        success: false,
        error: 'Meal plan is required',
      });
    }

    // Calculate alignment percentage based on goal
    const avgDailyCalories = plan.weeklyTotals.calories / 7;
    const avgDailyProtein = plan.weeklyTotals.protein / 7;

    let targetCalories = 2000;
    let targetProtein = 80;

    const goal = profile?.primaryGoal || 'general_health';
    if (goal === 'weight_loss') {
      targetCalories = 1600;
      targetProtein = 100;
    } else if (goal === 'muscle_gain') {
      targetCalories = 2500;
      targetProtein = 150;
    }

    const calorieAlignment = Math.max(0, 100 - Math.abs(avgDailyCalories - targetCalories) / targetCalories * 100);
    const proteinAlignment = Math.max(0, 100 - Math.abs(avgDailyProtein - targetProtein) / targetProtein * 100);
    const alignment = Math.round((calorieAlignment + proteinAlignment) / 2);

    const mlInfo = mlMetrics 
      ? ` (ML Model: ${mlMetrics.model}, F1: ${(mlMetrics.metrics.test.f1_macro * 100).toFixed(1)}%)`
      : '';
    
    const systemPrompt = `You are a nutrition analyst powered by machine learning${mlInfo}. Analyze meal plans and provide actionable suggestions. Respond with JSON only.`;

    const planSummary = plan.days.map((d: any) => 
      `${d.day}: ${d.meals.map((m: any) => `${m.type}=${m.name} (${m.calories}kcal)`).join(', ')}`
    ).join('\n');

    const userPrompt = `Analyze this 7-day meal plan for a user with goal: ${goal}

${planSummary}

Weekly Totals: ${plan.weeklyTotals.calories}kcal, ${plan.weeklyTotals.protein}g protein, $${plan.weeklyTotals.estimatedCost}

Provide 3 specific suggestions to optimize this plan. Output JSON:
{
  "summary": "Brief 1-2 sentence overview of plan quality",
  "suggestions": [
    {
      "type": "swap",
      "meal": "Day X Mealtype",
      "alt": "Alternative meal name with macros",
      "reason": "Why this swap helps the goal"
    }
  ]
}`;

    // Ensure model is warmed up
    await warmUpModel();
    
    const response = await ollama.chat({
      model: getMealPlanModel(),
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt },
      ],
      options: { 
        num_predict: 500,  // Reduced for speed
        temperature: 0.4,  // Lower for consistency
        num_ctx: 1024,     // Smaller context
        top_k: 15,
        top_p: 0.8,
      },
    });

    const content = response?.message?.content || '';
    const jsonMatch = content.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
    const jsonString = jsonMatch ? jsonMatch[1] : content;
    const insights = JSON.parse(jsonString.trim());

    res.json({
      success: true,
      insight: {
        alignment,
        summary: insights.summary || 'Your plan looks good overall.',
        suggestions: insights.suggestions || [],
      },
    });
  } catch (error: any) {
    console.error('[MealPlan] Insights error:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'Failed to generate insights',
    });
  }
};

// Generate shopping list from plan
export const generateShoppingList = async (req: Request, res: Response) => {
  try {
    const { plan, pantryItems } = req.body;

    if (!plan || !plan.days) {
      return res.status(400).json({
        success: false,
        error: 'Meal plan is required',
      });
    }

    const allIngredients = plan.days.flatMap((day: any) =>
      day.meals.flatMap((meal: any) => meal.ingredients || [])
    );

    const systemPrompt = `You are a grocery list organizer. Consolidate ingredients into categories and estimate costs. Respond with JSON only.`;

    const userPrompt = `Create a consolidated shopping list from these ingredients:
${JSON.stringify(allIngredients)}

Pantry items to exclude: ${JSON.stringify(pantryItems || [])}

Group by category (vegetables, proteins, grains, dairy, etc.), consolidate duplicates (e.g., "chicken x3 meals" → "3 chicken breasts"), estimate total cost.

Output JSON:
{
  "categories": {
    "Vegetables": ["spinach 500g", "tomatoes 6pc"],
    "Proteins": ["chicken breast 3pc", "..."],
    "Grains": ["..."],
    "Dairy": ["..."],
    "Other": ["..."]
  },
  "totalCost": 45.50
}`;

    // Ensure model is warmed up
    await warmUpModel();
    
    const response = await ollama.chat({
      model: getMealPlanModel(),
      messages: [
        { role: 'system', content: systemPrompt },
        { role: 'user', content: userPrompt },
      ],
      options: { 
        num_predict: 600,  // Reduced for speed
        temperature: 0.3,  // Lower for consistency
        num_ctx: 1024,     // Smaller context
        top_k: 15,
        top_p: 0.8,
      },
    });

    const content = response?.message?.content || '';
    const jsonMatch = content.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
    const jsonString = jsonMatch ? jsonMatch[1] : content;
    const shoppingList = JSON.parse(jsonString.trim());

    res.json({
      success: true,
      shoppingList,
    });
  } catch (error: any) {
    console.error('[MealPlan] Shopping list error:', error);
    res.status(500).json({
      success: false,
      error: error.message || 'Failed to generate shopping list',
    });
  }
};
